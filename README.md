# CUDA 性能调优实战指南

> 从零到独立调优 AI Kernel 的系统学习路径

## 关于本书

本书是一份面向 **深度学习 / AI 方向** 的 CUDA 性能调优实战教程。目标是让读者在面对一个 AI 算子 kernel 时，能够：

1. 用 Profiler（Nsight Compute）判断它是 **访存瓶颈** 还是 **计算瓶颈**
2. 找到具体的限制因素（访存合并、occupancy、bank conflict、指令吞吐等）
3. 应用正确的优化手段，并用数据验证提升

## 硬件环境

| 卡 | 架构 / arch | FP32 峰值 | 显存带宽 | 脊点（FLOP/Byte） | 角色 |
|---|---|---|---|---|---|
| **A100** | Ampere / sm_80 | ≈19.5 TFLOP/s | ≈1.5–2.0 TB/s | ≈10–13 | 最终验证 / 数据中心目标卡 |
| **3060 Laptop** | Ampere / sm_86 | ≈10–13 TFLOP/s | ≈336 GB/s | ≈34.5 | 日常迭代台 |

## 技能路线

1. 性能心智模型：Roofline、访存 vs 计算瓶颈
2. 访存优化：coalescing、shared memory、bank conflict
3. 执行配置：occupancy、launch 配置、ILP
4. Profiling 实操：Nsight Compute 的 Speed of Light / Memory Workload
5. AI 算子实战：reduction、GEMM tiling、softmax/attention

---

开始阅读：[第 1 节 · Roofline 心智模型](learning-records/0001-kickoff-mission-and-first-lesson.md)
