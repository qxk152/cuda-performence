# Mission: 独立调优 CUDA Kernel(AI 方向)

## The Why
用户想系统掌握 CUDA 性能调优,**能独立地把一个慢的 kernel 优化到接近硬件极限**。
应用方向是**深度学习 / AI**:优化神经网络算子、推理/训练 kernel(如 GEMM、attention、归一化、逐元素算子等)。

## The Goal
学完后,用户面对一个 AI 算子 kernel 时,能够:
1. 用 profiler(Nsight Compute)判断它是 **访存瓶颈** 还是 **计算瓶颈**。
2. 找到具体的限制因素(访存合并、occupancy、bank conflict、指令吞吐等)。
3. 应用正确的优化手段,并用数据验证提升。

## Context / Constraints
- **基础**:写过一些 CUDA kernel,理解 grid / block / thread 模型,但没系统调过性能。
- **硬件**:**NVIDIA A100(Ampere,compute capability 8.0)**。峰值 FP32 ≈ 19.5 TFLOP/s,HBM2e 带宽 ≈ 1.5–2.0 TB/s,脊点 ≈ 10–13 FLOP/Byte。可实际运行 nvcc 与 Nsight。
- **方向**:深度学习算子优化。
- **学习偏好**:见 [NOTES.md](./NOTES.md)。

## North Star 技能链(粗略路线)
1. 性能心智模型:roofline、访存 vs 计算瓶颈(本节起步)
2. 访存优化:coalescing、shared memory、bank conflict
3. 执行配置:occupancy、launch 配置、ILP
4. Profiling 实操:Nsight Compute 的 Speed of Light / Memory Workload
5. AI 算子实战:reduction、GEMM tiling、softmax/attention

## 状态
- 创建于:2025-06-12
- 当前阶段:刚启动,第 1 节课聚焦"建立 roofline 性能心智模型"。
