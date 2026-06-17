// 0008-reduce.cu — A100 reduction 逐步调优阶梯(对应 lessons/0008-reduction-optimization.html)
//
// 把一个大数组求和成一个标量。reduction 是访存瓶颈算子(算术强度 ~0.25 FLOP/Byte),
// 唯一目标是把 HBM 带宽吃满。本文件按经典教学路线(Mark Harris reduce0→reduce7)
// 把优化拆成一级一级的独立 kernel,每一刀只治一个病,benchmark 里逐版对照看加速比:
//
//   v0 interleaved+modulo   —— 教科书 baseline:取模发散 + bank conflict(最慢)
//   v1 interleaved 无发散    —— 去掉取模,warp 内不再发散;但跨步索引仍撞 bank conflict
//   v2 sequential addressing —— 反转循环方向:活跃线程连续 + shared 地址连续 → 无 bank conflict
//   v3 first-add-on-load     —— 加载即相加:grid 减半,一半线程不再一上来就空转
//   v4 unroll last warp      —— 最后一个 warp 用 __shfl_down_sync,省掉最后 5 轮 shared 往返与 sync
//   v5 grid-stride 多元素    —— 每线程先吞 N/总线程数 个元素:grid 与 N 解耦(百万→几百)+ 摊薄开销 + 全程合并
//   final = v5 + warp shuffle + atomicAdd 收尾(grid 很小,一趟搞定)
//
// v0..v4 走"大 grid、每线程 1(或 2)元素",暴露块内归约的病;
// v5/final 走"小 grid + grid-stride"。注意:grid-stride 的领先要在带宽墙高的卡(A100)上才明显——
// 消费卡(3060 ~330 GB/s)v4 已撞带宽墙,v5 与之打平属正常,见 main 末尾说明。
//
// 编译:  nvcc -O3 -arch=sm_80 0008-reduce.cu -o reduce      # A100
//         nvcc -O3 -arch=sm_86 0008-reduce.cu -o reduce      # 3060 Laptop
// 运行:  ./reduce
// Profile: ncu --set full --section MemoryWorkloadAnalysis ./reduce   # 原生 Linux;WSL 下 ncu 内核 profiling 不可用

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK 256

// ---- 错误检查宏 ----
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// warp 内归约:最后 32 个 lane 在寄存器里完成,不碰 shared,无需 __syncthreads
//   mask = 0xffffffff:本 warp 32 个 lane 全部参与(都活跃)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warpReduceSum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// ===========================================================================
// v0 —— interleaved addressing + 取模筛线程(教科书 baseline,两个病齐全)
//   ① tid % (2*s) == 0:一个 warp 里只有零星 lane 干活,其余空转却仍占调度 → 分支发散;
//   ② sdata[tid + s] 跨步访问,在 shared 里撞 bank conflict。
//   每线程只读 1 个元素 → 需要 G = ceil(N/BLOCK) 个 block(百万级)。
// ===========================================================================
__global__ void reduceV0_interleavedModulo(const float* __restrict__ g_in,
                                            float* __restrict__ g_out,
                                            size_t n) {
    __shared__ float sdata[BLOCK];
    const int tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * BLOCK + tid;

    sdata[tid] = (idx < n) ? g_in[idx] : 0.0f;   // 每线程 1 元素,无 grid-stride
    __syncthreads();

    for (int s = 1; s < BLOCK; s *= 2) {
        if (tid % (2 * s) == 0)                  // 病①:取模发散树
            sdata[tid] += sdata[tid + s];        // 病②:跨步 shared → bank conflict
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(g_out, sdata[0]);
}

// ===========================================================================
// v1 —— interleaved addressing,去发散(治病①,病②仍在)
//   不再用取模,而是把活跃线程压到 warp 前部:index = 2*s*tid。
//   小步长时,活跃 lane 在 warp 内连续 → warp 不再发散;
//   但 index 与 index+s 仍是跨步访问 → bank conflict 没治。
// ===========================================================================
__global__ void reduceV1_interleavedNoDiverge(const float* __restrict__ g_in,
                                              float* __restrict__ g_out,
                                              size_t n) {
    __shared__ float sdata[BLOCK];
    const int tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * BLOCK + tid;

    sdata[tid] = (idx < n) ? g_in[idx] : 0.0f;
    __syncthreads();

    for (int s = 1; s < BLOCK; s *= 2) {
        int index = 2 * s * tid;                 // 活跃线程连续映射 → 无发散
        if (index < BLOCK)
            sdata[index] += sdata[index + s];    // 仍跨步 → bank conflict
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(g_out, sdata[0]);
}

// ===========================================================================
// v2 —— sequential addressing(第一刀:反转循环方向,一石二鸟)
//   从大步长往小走,活跃线程永远是连续的前一半 tid<s。
//   ① 同一 warp 要么全活跃要么全休眠 → 无发散;
//   ② sdata[tid] 与 sdata[tid+s] 都是连续地址 → 无 bank conflict。
// ===========================================================================
__global__ void reduceV2_sequential(const float* __restrict__ g_in,
                                    float* __restrict__ g_out,
                                    size_t n) {
    __shared__ float sdata[BLOCK];
    const int tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * BLOCK + tid;

    sdata[tid] = (idx < n) ? g_in[idx] : 0.0f;
    __syncthreads();

    for (int s = BLOCK / 2; s > 0; s >>= 1) {    // 反转方向:活跃线程连续
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(g_out, sdata[0]);
}

// ===========================================================================
// v3 —— first add during load(加载即相加:消掉"上来就一半空转")
//   v2 第一轮就有一半线程闲着。让每个线程在写 shared 前先读 2 个元素加起来,
//   于是 grid 减半(每 block 覆盖 2*BLOCK 个元素),固定开销摊薄一倍。
//   注意:本版 grid = ceil(N / (2*BLOCK))。
// ===========================================================================
__global__ void reduceV3_firstAddOnLoad(const float* __restrict__ g_in,
                                        float* __restrict__ g_out,
                                        size_t n) {
    __shared__ float sdata[BLOCK];
    const int tid = threadIdx.x;
    size_t base = (size_t)blockIdx.x * (BLOCK * 2) + tid;

    float a = (base < n)         ? g_in[base]         : 0.0f;
    float b = (base + BLOCK < n) ? g_in[base + BLOCK] : 0.0f;
    sdata[tid] = a + b;                          // 加载即相加,合并访问(两段都连续)
    __syncthreads();

    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(g_out, sdata[0]);
}

// ===========================================================================
// v4 —— unroll last warp(第二刀:最后一个 warp 用 warp shuffle)
//   循环走到 s <= 32 只剩一个 warp,还用 shared + __syncthreads 纯属浪费。
//   降到 32 就停手,改用 __shfl_down_sync 在寄存器里做完最后 5 轮。
//   grid 同 v3(加载即相加 + 收尾洗牌)。
// ===========================================================================
__global__ void reduceV4_unrollWarp(const float* __restrict__ g_in,
                                    float* __restrict__ g_out,
                                    size_t n) {
    __shared__ float sdata[BLOCK];
    const int tid = threadIdx.x;
    size_t base = (size_t)blockIdx.x * (BLOCK * 2) + tid;

    float a = (base < n)         ? g_in[base]         : 0.0f;
    float b = (base + BLOCK < n) ? g_in[base + BLOCK] : 0.0f;
    sdata[tid] = a + b;
    __syncthreads();

    for (int s = BLOCK / 2; s > 32; s >>= 1) {   // 降到 32 就停,把最后一个 warp 交给寄存器
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        float v = sdata[tid];
        if (BLOCK >= 64) v += sdata[tid + 32];   // 把 32..63 合进来
        v = warpReduceSum(v);                    // 寄存器洗牌,零 shared、零 sync
        if (tid == 0)
            atomicAdd(g_out, v);
    }
}

// ===========================================================================
// v5 / final —— grid-stride 多元素(第三刀,最关键)+ 顺序归约 + warp shuffle + atomicAdd
//   每个线程先用 grid-stride 循环把 N/总线程数 个元素在寄存器里加完,再进 shared。
//   stride = gridDim.x * BLOCK 保证同一拍 warp 内 32 线程读连续 32 个 float → 全程合并。
//   grid 取 SM 数的若干倍即可(小 grid),固定开销被彻底摊薄,DRAM 带宽吃到 80–90%。
// ===========================================================================
__global__ void reduceFinal_gridStride(const float* __restrict__ g_in,
                                       float* __restrict__ g_out,
                                       size_t n) {
    __shared__ float sdata[BLOCK];
    const int tid = threadIdx.x;

    // (1) grid-stride load:每线程把多个元素先在寄存器里加起来,始终合并访问。
    float sum = 0.0f;
    size_t i      = (size_t)blockIdx.x * BLOCK + tid;
    size_t stride = (size_t)gridDim.x * BLOCK;
    for (; i < n; i += stride)
        sum += g_in[i];

    sdata[tid] = sum;
    __syncthreads();

    // (2) 顺序寻址块内归约,降到 32 就停。
    for (int s = BLOCK / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // (3) 最后一个 warp 寄存器洗牌;(4) 0 号 lane atomicAdd 收尾。
    if (tid < 32) {
        float v = sdata[tid];
        if (BLOCK >= 64) v += sdata[tid + 32];
        v = warpReduceSum(v);
        if (tid == 0)
            atomicAdd(g_out, v);
    }
}

// ---------------------------------------------------------------------------
// 跑一个 kernel:预热 + 多次计时取平均,校验结果,打印带宽。返回 ms/次。
// ---------------------------------------------------------------------------
typedef void (*ReduceFn)(const float*, float*, size_t);

static float benchReduce(const char* name, ReduceFn kernel,
                         int grid, const float* d_in, float* d_out,
                         size_t n, size_t bytes) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热一次(首启含 JIT/缓存效应,不计入)
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    kernel<<<grid, BLOCK>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iters = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        kernel<<<grid, BLOCK>>>(d_in, d_out, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
    float ms = ms_total / iters;

    float h_out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    double expected = (double)n;                  // 全 1 求和
    double rel_err  = fabs((double)h_out - expected) / expected;
    double gbps     = (double)bytes / (ms * 1e-3) / 1e9;

    printf("[%-22s] 耗时 %.3f ms/次  带宽 %7.1f GB/s  结果 %.0f  误差 %.2e  %s\n",
           name, ms, gbps, h_out, rel_err, rel_err < 1e-2 ? "PASS" : "FAIL");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main() {
    // N = 2^28 ≈ 2.68 亿个 float ≈ 1 GB,正好对上课里算的理论下限例子。
    const size_t N = 1ull << 28;
    const size_t bytes = N * sizeof(float);

    printf("N = %zu (%.2f GB),BLOCK = %d\n", N, bytes / 1e9, BLOCK);

    // ---- host 数据:全填 1.0f,期望和 = N,便于校验 ----
    float* h_in = (float*)malloc(bytes);
    for (size_t i = 0; i < N; ++i) h_in[i] = 1.0f;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // ---- 各版本的 launch config ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int gridSmall = prop.multiProcessorCount * 32;            // grid-stride 版:小 grid(几百~上千)
    int gridFull  = (int)((N + BLOCK - 1) / BLOCK);           // v0..v2:每线程 1 元素 → 百万级 block
    int gridHalf  = (int)((N + 2 * BLOCK - 1) / (2 * BLOCK)); // v3/v4:加载即相加 → grid 减半

    printf("grid: v0..v2=%d  v3/v4=%d  v5/final=%d  (SM=%d)\n",
           gridFull, gridHalf, gridSmall, prop.multiProcessorCount);
    printf("----------------------------------------------------------------\n");

    // ---- 逐版调优阶梯,看每一刀的加速 ----
    float ms0 = benchReduce("v0 interleaved+modulo", reduceV0_interleavedModulo,   gridFull,  d_in, d_out, N, bytes);
    float ms1 = benchReduce("v1 interleaved noDiv",  reduceV1_interleavedNoDiverge, gridFull,  d_in, d_out, N, bytes);
    float ms2 = benchReduce("v2 sequential addr",    reduceV2_sequential,           gridFull,  d_in, d_out, N, bytes);
    float ms3 = benchReduce("v3 firstAddOnLoad",     reduceV3_firstAddOnLoad,       gridHalf,  d_in, d_out, N, bytes);
    float ms4 = benchReduce("v4 unroll last warp",   reduceV4_unrollWarp,           gridHalf,  d_in, d_out, N, bytes);
    float ms5 = benchReduce("v5/final grid-stride",  reduceFinal_gridStride,        gridSmall, d_in, d_out, N, bytes);

    printf("----------------------------------------------------------------\n");
    printf("逐刀加速比(相对上一版):\n");
    printf("  v0→v1 去发散      %.2fx\n", ms0 / ms1);
    printf("  v1→v2 顺序寻址    %.2fx\n", ms1 / ms2);
    printf("  v2→v3 加载即相加  %.2fx\n", ms2 / ms3);
    printf("  v3→v4 warp shuffle %.2fx\n", ms3 / ms4);
    printf("  v4→v5 grid-stride %.2fx\n", ms4 / ms5);
    printf("总加速比 v0→final:  %.2fx\n", ms0 / ms5);
    printf("----------------------------------------------------------------\n");
    printf("注:grid-stride(v5)真正的价值是把 block 数从百万级压到几百(grid 与 N 解耦),\n");
    printf("    并摊薄固定启动/归约开销。但一旦 baseline 已撞带宽墙,它与大 grid 版打平——\n");
    printf("    消费卡(3060 ~330 GB/s)上 v4 就已吃满,v5 持平属正常;\n");
    printf("    A100(~1.8 TB/s)带宽墙高得多,grid-stride 的领先才看得出来。\n");
    printf("(对照:本机峰值带宽看 GPU 规格;DRAM Throughput 上 80%%+ 即调到头)\n");

    // ---- 清理 ----
    free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
