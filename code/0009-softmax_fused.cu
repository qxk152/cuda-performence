// 0009-softmax_fused.cu — 融合 softmax vs 4-kernel 基线对照
// 对应 lessons/0009-softmax-layernorm-fusion.html
//
// 输入 [rows x N],对每一行做数值稳定 softmax: y_i = exp(x_i - max) / Σ exp(x_j - max)
// 比较两种实现的等效 HBM 带宽:
//   baseline : 4 个 kernel(max / exp / sum / div),中间量落 HBM,流量 ~6N·rows
//   fused    : 每行一个 block,整行进 shared,online 归约 + warp shuffle,流量 ~2N·rows
//
// 编译:  nvcc -O3 -arch=sm_86 0009-softmax_fused.cu -o softmax   (A100 用 sm_80)
// 运行:  ./softmax
// 验证靠 CUDA event 计时 + 手算等效带宽(WSL 下 ncu 不可用)。

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ===========================================================================
// 基线:4 个独立 kernel。每一趟都把整行从 HBM 读出 / 写回,中间量 e 落显存。
// 一行一个 block,block 内用 shared 做朴素归约(够公平,瓶颈在 HBM 往返不在归约树)。
// ===========================================================================

// 趟1:m[r] = max_j x[r,j]
__global__ void k_rowmax(const float* __restrict__ x, float* __restrict__ m,
                         int N) {
    extern __shared__ float sm[];
    int r = blockIdx.x, tid = threadIdx.x;
    const float* row = x + (size_t)r * N;
    float v = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) v = fmaxf(v, row[i]);
    sm[tid] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] = fmaxf(sm[tid], sm[tid + s]);
        __syncthreads();
    }
    if (tid == 0) m[r] = sm[0];
}

// 趟2:e[r,j] = exp(x[r,j] - m[r])   —— 读 x 写 e
__global__ void k_exp(const float* __restrict__ x, const float* __restrict__ m,
                      float* __restrict__ e, int N) {
    int r = blockIdx.x, tid = threadIdx.x;
    float mr = m[r];
    const float* xr = x + (size_t)r * N;
    float* er = e + (size_t)r * N;
    for (int i = tid; i < N; i += blockDim.x) er[i] = __expf(xr[i] - mr);
}

// 趟3:s[r] = Σ_j e[r,j]   —— 读 e
__global__ void k_rowsum(const float* __restrict__ e, float* __restrict__ s,
                         int N) {
    extern __shared__ float sm[];
    int r = blockIdx.x, tid = threadIdx.x;
    const float* row = e + (size_t)r * N;
    float v = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) v += row[i];
    sm[tid] = v; __syncthreads();
    for (int t = blockDim.x / 2; t > 0; t >>= 1) {
        if (tid < t) sm[tid] += sm[tid + t];
        __syncthreads();
    }
    if (tid == 0) s[r] = sm[0];
}

// 趟4:y[r,j] = e[r,j] / s[r]   —— 读 e 写 y
__global__ void k_div(const float* __restrict__ e, const float* __restrict__ s,
                      float* __restrict__ y, int N) {
    int r = blockIdx.x, tid = threadIdx.x;
    float inv = 1.0f / s[r];
    const float* er = e + (size_t)r * N;
    float* yr = y + (size_t)r * N;
    for (int i = tid; i < N; i += blockDim.x) yr[i] = er[i] * inv;
}

// ===========================================================================
// 融合:每行一个 block,整行进 shared。online 归约同时拿 (max,sum),warp shuffle 收尾。
// HBM 上 x 只读一次、y 只写一次 ≈ 2N·rows。
// ===========================================================================
__global__ void k_softmax_fused(const float* __restrict__ x,
                                float* __restrict__ y, int N) {
    extern __shared__ float srow[];           // 大小 = N * sizeof(float)
    __shared__ float warpM[32];               // 每个 warp 的局部 max
    __shared__ float warpS[32];               // 每个 warp 的局部 sum(已对齐到该 warp 的 max)
    int tid = threadIdx.x;
    const float* row = x + (size_t)blockIdx.x * N;
    float* out = y + (size_t)blockIdx.x * N;

    // 趟1:读 HBM → shared,同时 online 累积本线程的 (m,s)
    float m = -FLT_MAX, s = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = row[i];
        srow[i] = v;
        float m_new = fmaxf(m, v);
        s = s * __expf(m - m_new) + __expf(v - m_new);
        m = m_new;
    }

    // warp 内对 (m,s) 二元组做 online 归约
    unsigned mask = 0xffffffff;
    for (int off = 16; off > 0; off >>= 1) {
        float m2 = __shfl_down_sync(mask, m, off);
        float s2 = __shfl_down_sync(mask, s, off);
        float M = fmaxf(m, m2);
        s = s * __expf(m - M) + s2 * __expf(m2 - M);
        m = M;
    }
    int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) { warpM[wid] = m; warpS[wid] = s; }
    __syncthreads();

    // 第一个 warp 把各 warp 的 (m,s) 再归约一次,得到全行 (M,S)
    int nwarps = (blockDim.x + 31) >> 5;
    if (wid == 0) {
        m = (lane < nwarps) ? warpM[lane] : -FLT_MAX;
        s = (lane < nwarps) ? warpS[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            float m2 = __shfl_down_sync(mask, m, off);
            float s2 = __shfl_down_sync(mask, s, off);
            float M = fmaxf(m, m2);
            s = s * __expf(m - M) + s2 * __expf(m2 - M);
            m = M;
        }
        if (lane == 0) { warpM[0] = m; warpS[0] = s; }
    }
    __syncthreads();

    float M = warpM[0], invS = 1.0f / warpS[0];
    // 趟2:从 shared 读,算 exp 并归一化,写 y。HBM 不再碰 x。
    for (int i = tid; i < N; i += blockDim.x)
        out[i] = __expf(srow[i] - M) * invS;
}

// ===========================================================================
// host 参考实现 + 计时驱动
// ===========================================================================
static void cpu_softmax(const float* x, float* y, int rows, int N) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * N;
        float* yr = y + (size_t)r * N;
        float m = -FLT_MAX;
        for (int i = 0; i < N; ++i) m = fmaxf(m, xr[i]);
        double s = 0.0;
        for (int i = 0; i < N; ++i) s += expf(xr[i] - m);
        for (int i = 0; i < N; ++i) yr[i] = expf(xr[i] - m) / (float)s;
    }
}

static double max_abs_err(const float* a, const float* b, size_t n) {
    double e = 0.0;
    for (size_t i = 0; i < n; ++i) e = fmax(e, fabs((double)a[i] - b[i]));
    return e;
}

int main() {
    const int rows = 8192;     // 行数
    const int N    = 4096;     // 每行长度(整行 4096*4=16KB 可进 shared)
    const int BLK  = 256;
    const size_t cnt   = (size_t)rows * N;
    const size_t bytes = cnt * sizeof(float);
    printf("rows=%d  N=%d  总数据 %.2f MB\n", rows, N, bytes / 1e6);

    // host 数据:随机
    float* h_x = (float*)malloc(bytes);
    float* h_y = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    srand(123);
    for (size_t i = 0; i < cnt; ++i) h_x[i] = (float)(rand() % 2000 - 1000) / 100.0f;
    cpu_softmax(h_x, h_ref, rows, N);

    float *d_x, *d_y, *d_e, *d_m, *d_s;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_e, bytes));                 // 基线的中间量 e
    CUDA_CHECK(cudaMalloc(&d_m, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    const int iters = 20;
    size_t shmem = (size_t)BLK * sizeof(float);          // 基线归约用
    size_t shrow = (size_t)N * sizeof(float);            // 融合版整行用

    // ---- 基线:4 kernel ----
    auto run_baseline = [&]() {
        k_rowmax<<<rows, BLK, shmem>>>(d_x, d_m, N);
        k_exp   <<<rows, BLK>>>(d_x, d_m, d_e, N);
        k_rowsum<<<rows, BLK, shmem>>>(d_e, d_s, N);
        k_div   <<<rows, BLK>>>(d_e, d_s, d_y, N);
    };
    run_baseline(); CUDA_CHECK(cudaDeviceSynchronize());     // 预热
    CUDA_CHECK(cudaEventRecord(a));
    for (int it = 0; it < iters; ++it) run_baseline();
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    float ms_base; CUDA_CHECK(cudaEventElapsedTime(&ms_base, a, b)); ms_base /= iters;
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    double err_base = max_abs_err(h_y, h_ref, cnt);

    // ---- 融合 ----
    k_softmax_fused<<<rows, BLK, shrow>>>(d_x, d_y, N);      // 预热
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(a));
    for (int it = 0; it < iters; ++it)
        k_softmax_fused<<<rows, BLK, shrow>>>(d_x, d_y, N);
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    float ms_fused; CUDA_CHECK(cudaEventElapsedTime(&ms_fused, a, b)); ms_fused /= iters;
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    double err_fused = max_abs_err(h_y, h_ref, cnt);

    // ---- 等效带宽:用各自的理论 HBM 流量 ----
    double gb_base  = 6.0 * bytes;   // x读2 + e写1读2 + y写1 ≈ 6N
    double gb_fused = 2.0 * bytes;   // x读1 + y写1 = 2N
    auto bw = [](double gbytes, float ms){ return gbytes / (ms * 1e-3) / 1e9; };

    printf("[baseline 4-kernel] %.3f ms  等效带宽 %.0f GB/s  maxErr %.2e\n",
           ms_base, bw(gb_base, ms_base), err_base);
    printf("[fused]             %.3f ms  等效带宽 %.0f GB/s  maxErr %.2e\n",
           ms_fused, bw(gb_fused, ms_fused), err_fused);
    printf("加速比 = %.2fx   (理论流量比 6N/2N = 3x)\n", ms_base / ms_fused);
    printf("正确性:两者 maxErr 都应 < 1e-5  → %s\n",
           (err_base < 1e-5 && err_fused < 1e-5) ? "PASS" : "FAIL");

    free(h_x); free(h_y); free(h_ref);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_e); cudaFree(d_m); cudaFree(d_s);
    return 0;
}
