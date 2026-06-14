# 0005 · 第 5 节课:Bank Conflict + 矩阵乘 Tiling(里程碑)

- **日期**: 2025-06-12
- **状态**: 进行中

## 第 5 节内容(综合里程碑)
把前四节拧进一个真实算子(GEMM)的完整优化。
- **Bank conflict**:shared memory 32 个 bank(收银台比方),`bank=(地址/4)%32`;
  不同地址同 bank → N 路冲突慢 N 倍;全员同址=broadcast 不算冲突。
  经典坑:按列读 `tile[32][32]` → 32 路冲突;修复 padding `tile[32][33]`。
- **Tiled GEMM**:朴素版(全局反复读,访存瓶颈)→ tiled 版(分块 shared 复用 T 次,
  全局访存降 1/T,roofline 右移逼近峰值)。
- **概念回收表**:roofline/读报告/合并/shared/bank 五节全在这一个 kernel 现身。
  明确点出"独立调优 kernel"的完整链路(测量→定位→查模式→找复用→修冲突→再测)。
- 练习**交错**前几节概念(bank + 访存瓶颈 + roofline 移动),检验真懂。

## 衔接 / 进度
- 这是 mission 目标"独立调优 kernel"的第一个完整缩影。访存优化主线基本讲完。
- 下一节转向**执行配置**:occupancy 与 launch 配置(藏延迟),对应第 2 节的"延迟瓶颈"。
- 速查表新增 Bank Conflict 速记 + Tiled GEMM 要点。

## 备注
- 用户进度很快,连续推进 5 节,理解吸收良好(主动深挖底层)。
- 待用户接入 A100 后,GEMM 是绝佳的"真机从慢到快"实操项目(naive vs tiled 实测加速比)。
