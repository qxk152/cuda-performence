# Resources

记录用于支撑教学的高质量、高信任资源。永远不信任记忆,以这些为准。

## 一级资源(官方 / 最高信任)

### CUDA C++ Best Practices Guide
- **链接**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **类型**: 官方文档
- **覆盖**: 优化优先级(APOD)、访存合并、occupancy、指令优化。调优的权威圣经。
- **信任度**: ★★★★★

### Nsight Compute Profiling Guide
- **链接**: https://docs.nvidia.com/nsight-compute/ProfilingGuide
- **类型**: 官方文档
- **覆盖**: Speed of Light、roofline、Memory Workload Analysis,判断访存/计算瓶颈的核心。
- **信任度**: ★★★★★

### Architecture Tuning Guides(按卡选)
- **Ampere(A100,你的卡)**: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html  ← **精读这一份**
- Ada: https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
- Hopper (H100): https://docs.nvidia.com/cuda/archive/13.2.0/pdf/Hopper_Tuning_Guide.pdf
- **类型**: 官方文档
- **覆盖**: 架构相关调优(tensor core、L2、async copy)。你的卡是 A100,精读 Ampere 这份。
- **信任度**: ★★★★★

## 二级资源(高质量社区 / 课程)

### GPU MODE Lectures
- **链接**: https://github.com/gpu-mode/lectures
- **类型**: 课程(视频 + 代码 + 幻灯片)
- **覆盖**: 面向 AI 的 CUDA 优化实战,Lecture 8 是性能 checklist。最贴合 AI 方向。
- **信任度**: ★★★★☆

### Programming Massively Parallel Processors (PMPP), 4th/5th ed.
- **链接**: https://www.amazon.com/dp/0323912311 (4th)
- **类型**: 教科书
- **覆盖**: GPU 编程与优化的标准教材,GPU MODE 课程与之配套。
- **信任度**: ★★★★★

### GPU MODE 笔记(Christian Mills)
- **链接**: https://christianjmills.com/series/notes/cuda-mode-notes.html
- **类型**: 课程笔记
- **覆盖**: GPU MODE 各讲的文字版笔记,含 Lecture 8 性能 checklist。
- **信任度**: ★★★★☆

## 三级资源(实操教程)

### Using Nsight Compute to Inspect your Kernels (NVIDIA Blog)
- **链接**: https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/
- **类型**: 官方博客教程
- **覆盖**: Nsight Compute 上手与 guided analysis。

## 社区(获取智慧 / 实战检验)
- **GPU MODE Discord**: https://discord.gg/gpumode — 最活跃的 GPU kernel 优化社区,AI 方向首选。
- **r/CUDA**: https://www.reddit.com/r/CUDA/ — 提问与讨论。
- **NVIDIA Developer Forums (CUDA)**: https://forums.developer.nvidia.com/c/accelerated-computing/cuda/ — 官方论坛。

## 待办
- [x] 确认 GPU 型号 → A100 / Ampere / sm_80,已锁定 Ampere tuning guide。
