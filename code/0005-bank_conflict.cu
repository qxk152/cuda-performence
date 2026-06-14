// 0005-bank_conflict.cu — 实测 shared memory bank conflict 的代价
// 对应 lessons/0005-bank-conflict-and-tiling.html 第 3 节
//
// 同一个 warp 内,32 个线程按不同 stride 访问 shared:
//   stride=1  → 32 线程落 32 个不同 bank   → 无冲突,1 拍
//   stride=2  → 挤进 16 个 bank,各 2 线程  → 2 路冲突,~2 拍
//   stride=32 → 全落 bank 0 的不同字        → 32 路冲突,~32 拍
// 用 CUDA event 计时,看耗时是否随冲突路数线性变差。
//
// 编译: nvcc -O3 -arch=sm_86 0005-bank_conflict.cu -o bank   (A100 用 sm_80)
// 运行: ./bank

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e__ = (call);                                              \
        if (e__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(e__), __FILE__, __LINE__);             \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define BLK   256          // 每 block 线程数(= 8 个 warp)
#define SMEM  4096         // shared 容量(字),需 >= 最大下标 (BLK-1)*32 ... 用取模收回范围
#define REPS  4096         // 重复访问次数,放大 bank 冲突的相对差异

// STRIDE 作为模板参数:编译期常量,循环里无分支开销,纯粹暴露访存代价。
template<int STRIDE>
__global__ void bankKernel(float* __restrict__ out) {
    __shared__ float s[SMEM];
    int tid = threadIdx.x;

    // 预填 shared(合并写,无冲突)
    for (int i = tid; i < SMEM; i += BLK) s[i] = (float)i;
    __syncthreads();

    // 关键循环:每个线程按 STRIDE 取地址,反复读累加。
    // 取模把下标收回 [0,SMEM),保证同一 warp 内 32 线程的 bank 关系由 STRIDE 决定。
    float acc = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < REPS; ++r) {
        int idx = ((tid * STRIDE) + r) & (SMEM - 1);   // SMEM 是 2 的幂,& 代替 %
        acc += s[idx];
    }
    // 写回 global,逼编译器不要丢弃上面的读
    out[blockIdx.x * BLK + tid] = acc;
}

template<int STRIDE>
static float timeKernel(const char* tag, float* d_out, int grid) {
    cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    bankKernel<STRIDE><<<grid, BLK>>>(d_out);                  // 预热
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    const int iters = 50;
    CUDA_CHECK(cudaEventRecord(a));
    for (int it = 0; it < iters; ++it) bankKernel<STRIDE><<<grid, BLK>>>(d_out);
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); ms /= iters;
    printf("  stride=%-2d  %s  : %.3f ms/次\n", STRIDE, tag, ms);
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
    return ms;
}

int main() {
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int grid = prop.multiProcessorCount * 8;   // 喂满机器即可
    printf("GPU: %s  (%d SM)  grid=%d block=%d reps=%d\n",
           prop.name, prop.multiProcessorCount, grid, BLK, REPS);

    float* d_out; CUDA_CHECK(cudaMalloc(&d_out, (size_t)grid * BLK * sizeof(float)));

    printf("\n按 stride 实测(同一 warp 32 线程的 bank 冲突路数 = gcd(stride,32) 决定):\n");
    float t1  = timeKernel<1>("无冲突      ", d_out, grid);
    float t2  = timeKernel<2>("2 路冲突    ", d_out, grid);
    float t4  = timeKernel<4>("4 路冲突    ", d_out, grid);
    float t8  = timeKernel<8>("8 路冲突    ", d_out, grid);
    float t16 = timeKernel<16>("16 路冲突   ", d_out, grid);
    float t32 = timeKernel<32>("32 路冲突   ", d_out, grid);
    float t33 = timeKernel<33>("33→无冲突   ", d_out, grid);  // 与 32 互质 → 散开

    printf("\n相对 stride=1 的倍率(理论 ≈ 冲突路数):\n");
    printf("  2:%.2fx  4:%.2fx  8:%.2fx  16:%.2fx  32:%.2fx  | 33:%.2fx(应≈1,印证互质消冲突)\n",
           t2/t1, t4/t1, t8/t1, t16/t1, t32/t1, t33/t1);

    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
