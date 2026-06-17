# 0003 · 第 3 节课:访存合并

- **日期**: 2025-06-12
- **状态**: 进行中

## 第 3 节内容
- 主题:**访存合并(memory coalescing)**——访存瓶颈的头号优化手段,用户第一次"动手让 kernel 变快"。
- 核心技能:盯住 warp 内相邻线程(threadIdx.x 差 1)在同一指令的地址差;
  差 1 元素 → 合并,差大 stride → 非合并。`threadIdx.x` 应乘在数据最低维。
- 讲了 warp(32 线程)与 128B 事务的关系,行主序矩阵的经典踩坑(row=tid.x vs col=tid.x)。
- Nsight 信号:sectors per request(理想≈4)。
- 4 道判断题(1D 合并 / stride 非合并 / 矩阵两种映射)。

## 教学法
- 仍是纸面学:代码片段 + 思维判断,无需运行。
- 把合并规则也加进了 reference/roofline-cheatsheet.html(可复用参考)。

## 用户主动深挖(良好信号)
用户在本节连续追问并要求"固化进课程",已逐一加入:
1. **「事务是什么」** → 课程新增 §2.5(货箱比方 + request/sector 术语对应)。
2. **对齐(alignment)** → 课程新增 §4.5(128B 边界、偏移翻倍、cudaMalloc 保证)。
3. **L1/L2 缓存补救非合并** → 课程新增 §4.6(Memory% vs DRAM% 的差异含义、别依赖缓存)。
全部同步进速查表。
→ 用户偏好「把追问的延伸知识固化进资料」,后续主动提供延伸点并询问是否固化。
→ 用户求知欲强、愿深入底层机制,可适当提高深度。

## 下一步
- 第 4 节:**shared memory**——数据被反复读时搬进片上缓存,绕开显存带宽。
  自然衔接:合并是"读得对",shared memory 是"少读"。
- 之后:bank conflict → occupancy → reduction → GEMM tiling。
