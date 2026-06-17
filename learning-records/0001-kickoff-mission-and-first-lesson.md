# 0001 · 启动:确立使命与第一节课

- **日期**: 2025-06-12
- **状态**: 已确立

## 背景 / 用户画像
- 基础:写过一些 CUDA kernel,懂 grid/block/thread 模型,但**没系统调过性能**。
- 方向:**深度学习 / AI** 算子优化。
- 硬件:本地 **NVIDIA A100(Ampere,compute capability 8.0 / sm_80)**。可跑 nvcc / Nsight。编译用 `nvcc -arch=sm_80`。
- 目标:**能独立调优 kernel**。

## 关键决定
1. **使命确立**为「独立调优 AI kernel」。详见 MISSION.md。
2. **第一节课选 Roofline 心智模型**,而非直接讲某个优化技巧。
   - 理由:用户会写 kernel 但没调过性能,缺的是「先判断瓶颈再优化」的根基。
     这是 zone of proximal development 里最该先补的一块——它决定后续所有优化的方向。
3. 因为有真实数据中心卡,课程包含可本地运行的 `ncu` 验证环节。

## 待解决 / 下一步
- ~~待确认 GPU 具体型号~~ → **已确认:A100 / Ampere / sm_80**。已锁定 Ampere tuning guide,后续练习用 A100 真实数字。
- 下一节:用 Nsight Compute 读 Speed of Light 报告,把 roofline 落到真实数字。

## 路线推演(可能调整)
roofline → Nsight 实操 → 访存合并 → shared memory/bank conflict → occupancy → reduction → GEMM tiling → softmax/attention
