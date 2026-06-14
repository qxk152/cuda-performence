# CUDA 性能调优实战指南

> 从零到独立调优 AI Kernel 的系统学习路径

## 关于本书

本书是一份面向 **深度学习 / AI 方向** 的 CUDA 性能调优实战教程。目标是让读者在面对一个 AI 算子 kernel 时，能够：

1. 用 Profiler（Nsight Compute）判断它是 **访存瓶颈** 还是 **计算瓶颈**
2. 找到具体的限制因素（访存合并、occupancy、bank conflict、指令吞吐等）
3. 应用正确的优化手段，并用数据验证提升

## 项目结构

```
cuda-performence/
├── lessons/           # 课程讲义（精心排版的 HTML 页面）
│   ├── 0001-roofline-mental-model.html        第01课 Roofline 心智模型
│   ├── 0002-reading-speed-of-light.html       第02课 读懂 Speed of Light
│   ├── 0003-memory-coalescing.html            第03课 访存合并
│   ├── 0004-shared-memory.html                第04课 共享内存
│   ├── 0005-bank-conflict-and-tiling.html     第05课 Bank 冲突与分块
│   ├── 0006-occupancy-and-launch-config.html  第06课 占用率与启动配置
│   ├── 0007-dual-gpu-roofline-hands-on.html   第07课 双卡 Roofline 实操
│   ├── 0008-reduction-optimization.html       第08课 规约优化
│   └── 0009-softmax-layernorm-fusion.html     第09课 Softmax/LayerNorm 融合
│
├── code/              # 课程配套的完整可编译 CUDA 源码
│   ├── 0008-reduce.cu             规约优化的四步迭代（对应第08课）
│   └── 0009-softmax_fused.cu      融合 softmax vs 多 kernel 对照（对应第09课）
│
├── learning-records/  # 学习笔记（Markdown）
│   ├── 0001-kickoff-mission-and-first-lesson.md
│   ├── 0002-lesson2-and-environment.md
│   ├── 0003-lesson3-coalescing.md
│   ├── 0004-lesson4-shared-memory.md
│   ├── 0005-lesson5-bank-conflict-tiling.md
│   ├── 0006-lesson6-occupancy.md
│   └── 0007-dual-gpu-roofline-handson.md
│
├── reference/         # 速查参考卡片
│   └── roofline-cheatsheet.html   Roofline 瓶颈判断速查表
│
├── MISSION.md         # 学习任务书
├── NOTES.md           # 通用笔记
├── RESOURCES.md       # 推荐资源汇总
├── SUMMARY.md         # GitBook 目录索引
├── book.json          # HonKit 构建配置
└── .gitbook.yaml      # GitBook 元数据
```

### 目录说明

| 目录 | 格式 | 说明 |
|------|------|------|
| `lessons/` | HTML | 精心排版的课程讲义，包含交互练习与代码示例，是本项目的核心内容 |
| `code/` | CUDA (.cu) | 课程配套的完整可编译源码，每份代码与对应课程编号一致 |
| `learning-records/` | Markdown | 学习过程笔记，记录环境搭建、实操踩坑等 |
| `reference/` | HTML | 速查卡片，用于快速回顾关键概念 |

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

## 在线阅读

网站地址：[https://cuda-performence.qxk1998.top](https://cuda-performence.qxk1998.top)

---

开始阅读：[第 1 节 · Roofline 心智模型](lessons/0001-roofline-mental-model.html)
