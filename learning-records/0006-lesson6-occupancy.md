# 0006 · 第 6 节课:Occupancy 与 Launch 配置

- **日期**: 2025-06-12
- **状态**: 进行中

## 第 6 节内容
回收第 2 节埋的"延迟瓶颈"(两个利用率都低)。从访存优化主线转向**执行配置**。
- **藏延迟机制**:GPU 不等,停顿 warp 切到就绪 warp;前提是有足够多 warp 可切换。
- **occupancy 定义**:活跃 warp ÷ SM 最大 warp(A100:64 warp/2048 线程)。
- **三种限制资源**:寄存器/线程、shared mem/block、block 大小。各自对应 Nsight 字段。
- **反直觉铁律(重点)**:occupancy 高不一定快(ILP 也能藏延迟;提它有代价如 spilling);
  但过低一定伤性能。→ 保下限、不强求上限,只在"延迟瓶颈+occupancy 低"时才提。
- **launch 配置实用建议**:block 取 32 倍数(128/256),grid ≥ SM 数,
  `cudaOccupancyMaxPotentialBlockSize()` 自动推荐;`-Xptxas -v` 查寄存器。
- 练习交错(延迟瓶颈诊断 / occupancy 正误 / 寄存器限制 / 计算瓶颈下提 occupancy 无益)。

## 进度
- 核心调优框架基本完整:roofline → 读报告 → 合并/对齐 → shared/bank → occupancy。
- 下一节计划:**间隔复习测验**(跨 6 节 retrieval practice),把流畅转留存。
- 之后可选实战支线:reduction 优化、softmax/attention(AI 算子),或待接入 A100 后真机实操。

## 教学法
- 用 WebFetch 核实了 occupancy 官方定义与"不是越高越好",未凭记忆。
- 同步进速查表(Occupancy 速记块)。
