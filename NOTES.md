# Notes

教学偏好与工作笔记。

## 用户画像
- 基础:写过一些 CUDA kernel,懂 grid/block/thread,但没系统调过性能。
- 方向:深度学习 / AI 算子优化。
- 语言:中文教学。
- 目标:能独立调优 kernel。

### 硬件:双卡设定(2025-06-13 更新)
用户有**两块卡**,都能交互式登录跑命令 → 教学从纸面升级为**真机实操 + 双卡对比**。

| 卡 | 架构 / arch | FP32 峰值 | 显存带宽 | 脊点(FLOP/Byte) | 角色 |
|---|---|---|---|---|---|
| **A100** | Ampere / **sm_80** | ≈19.5 TFLOP/s | ≈1.5–2.0 TB/s | ≈10–13 | 最终验证 / 数据中心目标卡 |
| **3060 Laptop** | Ampere / **sm_86** | ≈10–13 TFLOP/s† | ≈336 GB/s | ≈34.5† | 日常迭代台(随时可跑) |

- **3060 = 笔记本版(GN20-E3 / GA106)**:30 SM、3840 核、**6GB GDDR6 / 192-bit / ≈336 GB/s**。CUDA 11.7,`nvcc`+`ncu`(2022.2)均在。
- † 笔记本 3060 的 FP32 **随 TGP(60–130W)浮动**,同名机器能差 20–30% → **峰值与脊点须用 `bandwidthTest` + 实测算力现场测,别抄规格表**(第 7 节核心金句)。
- 编译:A100 用 `-arch=sm_80`,3060 用 `-arch=sm_86`(**两者不同,每节课须标清**)。
- **关键教学点**:3060 脊点(≈35)远右于 A100(≈12.6)→ **同一 kernel(如算术强度=20)在 A100 是计算瓶颈、在 3060 变访存瓶颈**。消费卡 vs 数据中心卡最值钱的直觉,作第 7 节核心。

## 教学偏好
- 课程用中文。
- 用户有真实 GPU(双卡),练习应尽量包含**可本地运行 / profile** 的环节。
- 偏实战(AI 方向),概念要落到具体算子上。
- 用户偏好「把追问的延伸知识固化进资料」(见 0003),主动提供延伸点并询问是否固化。求知欲强,可提高深度。

## 待确认
- 已确认 GPU:A100(sm_80) + **3060 Laptop GPU 6GB(sm_86)**,均可交互式跑命令。
- 3060 工具链已全确认:`nvcc` 11.7 + `ncu` 2022.2 均在。
- 若 3060 在 **WSL** 下(主机名 LAPTOP-…):`ncu` 硬件性能计数器**完全不可用**——
  实测(2026-06-14,lesson8 reduce.cu)报错 `Profiling is not supported on device 0 as it uses the
  Windows Subsystem for Linux (WSL)`,且 "No kernels were profiled"。**结论:WSL 下 ncu 内核级 profiling 一律采不到**,
  3060 上的验证改用 **CUDA event 计时 + 手算等效带宽**;全量 ncu profile 留给 A100(原生 Linux)。
- 笔记本 3060 实际 TGP 未知 → 峰值算力第 7 节用微基准实测确定。

## 工作笔记
- 已建立 mission、resources、notes。第 1–6 节已完成(纸面框架:roofline→读报告→合并/对齐→shared/bank→occupancy)。
- **环境升级(2025-06-13)**:从「本地无工具链、纸面学」→「双卡均可交互式跑命令,真机实操」。
  - `ncu` 在 3060 上已确认可用(2022.2)。消费卡跑 ncu 常需 **sudo 或开 perf counter 权限**,若报权限错有解法。
  - **更正(2026-06-14)**:3060 在 WSL 下,ncu 内核 profiling 实测**完全不可用**(见上"待确认"已落实)。3060 验证靠 event 计时 + 手算带宽。
- 第 8 节(reduction)已实操:`code/0008-reduce.cu` 三版对照(textbook / naive / optim),3060 sm_86 实测
  优化版 ~309 GB/s ≈ 峰值 92%;naive/optim 打平(带宽墙),book 慢 3.34x(暴露 grid-stride 才是大头)。
- 第 9 节(softmax/LayerNorm 融合)讲义已出:`lessons/0009-softmax-layernorm-fusion.html`。
  主线=减少 HBM 往返(未融合 6N → 融合 2N),online softmax + warp shuffle 归约二元组,串到 FlashAttention。
- 下一节:attention 全链 / FlashAttention 分块思想(本节融合 softmax 是其核心积木)。
