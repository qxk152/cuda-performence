// 0008-reduce.cu — 完整可编译的 A100 reduction
// 对应 lessons/0008-reduction-optimization.html
//
// 把一个大数组求和成一个标量。reduction 是访存瓶颈算子(算术强度 ~0.25 FLOP/Byte),
// 唯一目标是把 HBM 带宽吃满。本文件综合了一节课里的四步优化:
//   1) grid-stride load   —— 合并访存,且让 block 数 G 很小
//   2) 反转循环方向       —— 活跃线程连续 + shared 地址连续 → 无 bank conflict
//   3) 最后一个 warp 用 __shfl_down_sync —— 省掉最后 5 轮 shared 往返与 __syncthreads
//   4) atomicAdd 收尾     —— G 很小,一趟 kernel 搞定
//
// 编译:  nvcc -O3 -arch=sm_80 0008-reduce.cu -o reduce
// 运行:  ./reduce
// Profile: ncu --set full --section MemoryWorkloadAnalysis ./reduce

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
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warpReduceSum(float v) {
    // mask = 0xffffffff:本 warp 32 个 lane 全部参与(都活跃)
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// ---------------------------------------------------------------------------
// 【教科书 baseline · Mark Harris reduction #1】最朴素的样子:
//   每线程只读 1 个元素,不用 grid-stride → G 暴涨到 ~N/BLOCK(百万级 block)。
//   叠加取模发散树 + bank conflict。这是 slide 里所有优化的起点。
//   注意:它需要 G = ceil(N/BLOCK) 个 block,不能用前两版的小 grid。
// ---------------------------------------------------------------------------
__global__ void reduceTextbookKernel(const float* __restrict__ g_in,
                                     float* __restrict__ g_out,
                                     size_t n) {
    __shared__ float sdata[BLOCK];
    const int tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * BLOCK + tid;

    sdata[tid] = (idx < n) ? g_in[idx] : 0.0f;   // 每线程 1 元素,无 grid-stride
    __syncthreads();

    // 取模发散树 + 跨步 shared 访问(两个病齐全)
    for (int s = 1; s < BLOCK; s *= 2) {
        if (tid % (2 * s) == 0)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(g_out, sdata[0]);
}

// ---------------------------------------------------------------------------
// 【朴素版 · 反面教材】lesson8 第 2 节的两个病:
//   ① tid % (2*s) == 0:一个 warp 里只有零星 lane 干活,其余空转却仍占调度;
//   ② sdata[tid + s] 这种跨步访问,在 shared 里撞 bank conflict。
// 为公平对照,grid-stride load 与收尾保持一致,只把 block 内归约换成朴素写法。
// ---------------------------------------------------------------------------
__global__ void reduceNaiveKernel(const float* __restrict__ g_in,
                                  float* __restrict__ g_out,
                                  size_t n) {
    __shared__ float sdata[BLOCK];
    const int tid = threadIdx.x;

    float sum = 0.0f;
    size_t i      = (size_t)blockIdx.x * BLOCK + tid;
    size_t stride = (size_t)gridDim.x * BLOCK;
    for (; i < n; i += stride)
        sum += g_in[i];

    sdata[tid] = sum;
    __syncthreads();

    // 病灶:正向步长 + 取模筛线程,全程不停 __syncthreads
    for (int s = 1; s < BLOCK; s *= 2) {
        if (tid % (2 * s) == 0)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(g_out, sdata[0]);
}

// ---------------------------------------------------------------------------
// reduction kernel:每个 block 产出一个部分和,再 atomicAdd 累加到 g_out
// ---------------------------------------------------------------------------
__global__ void reduceKernel(const float* __restrict__ g_in,
                             float* __restrict__ g_out,
                             size_t n) {
    __shared__ float sdata[BLOCK];
    const int tid = threadIdx.x;

    // (1) grid-stride load:每个线程把多个元素先在寄存器里加起来。
    //     stride = gridDim.x * BLOCK,保证同一轮里连续的线程读连续的地址 → 合并访存。
    float sum = 0.0f;
    size_t i      = (size_t)blockIdx.x * BLOCK + tid;
    size_t stride = (size_t)gridDim.x * BLOCK;
    for (; i < n; i += stride)
        sum += g_in[i];

    sdata[tid] = sum;
    __syncthreads();

    // (2) 反转循环方向的 block 内归约,降到 32 时停手。
    //     if (tid < s) 让活跃线程连续;sdata[tid] 与 sdata[tid+s] 都是连续地址 → 无 bank conflict。
    for (int s = BLOCK / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // (3) 最后一个 warp 用寄存器洗牌,省掉最后 5 轮 shared 往返与同步。
    if (tid < 32) {
        float v = sdata[tid];
        if (BLOCK >= 64) v += sdata[tid + 32];   // 把 32..63 合进来(BLOCK>32 时)
        v = warpReduceSum(v);
        // (4) 0 号 lane 持有本 block 的部分和,atomicAdd 收尾。
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

    printf("[%-6s] 耗时 %.3f ms/次  带宽 %.1f GB/s  结果 %.0f  误差 %.2e  %s\n",
           name, ms, gbps, h_out, rel_err, rel_err < 1e-2 ? "PASS" : "FAIL");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main() {
    // N = 2^28 ≈ 2.68 亿个 float ≈ 1 GB,正好对上课里算的理论下限例子。
    const size_t N = 1ull << 28;
    const size_t bytes = N * sizeof(float);

    printf("N = %zu (%.2f GB)\n", N, bytes / 1e9);

    // ---- host 数据:全填 1.0f,期望和 = N,便于校验 ----
    float* h_in = (float*)malloc(bytes);
    for (size_t i = 0; i < N; ++i) h_in[i] = 1.0f;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // ---- launch config:grid-stride 下 G 取个适中值即可(几百~上千)。
    //      按 SM 数 * 每 SM 常驻 block 数估一个,够把机器喂满又让 G 很小。----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int grid = prop.multiProcessorCount * 32;   // 例如 A100 108 SM → ~3456,仍远小于 N

    // ---- 对照:教科书 baseline(大 grid) vs 朴素版 vs 优化版 ----
    int gridFull = (int)((N + BLOCK - 1) / BLOCK);   // 教科书版:每线程 1 元素 → 百万级 block
    printf("grid: textbook=%d  (naive/optim=%d)\n", gridFull, grid);

    float ms_book  = benchReduce("book",  reduceTextbookKernel, gridFull, d_in, d_out, N, bytes);
    float ms_naive = benchReduce("naive",  reduceNaiveKernel, grid, d_in, d_out, N, bytes);
    float ms_opt   = benchReduce("optim",  reduceKernel,      grid, d_in, d_out, N, bytes);
    printf("加速比:naive/optim=%.2fx   book/optim=%.2fx\n",
           ms_naive / ms_opt, ms_book / ms_opt);
    printf("(对照:本机峰值带宽看 GPU 规格;DRAM Throughput 上 80%%+ 即调到头)\n");

    // ---- 清理 ----
    free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
