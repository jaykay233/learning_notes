# Lab1 · 手写冗余算子消除 Pass

> 对应原课 Lab1。前置：[01](../01-cpp-foundations.md)、[03](../03-mlir-dialect-op-type.md)、[04](../04-mlir-pass-patterns.md)。**无需 NPU。**

---

## 1. 为什么做这个 Lab

昇腾 ATC 的融合/消除是黑盒；本 Lab 用 MLIR 把 **同一类 Frontend 优化** 白盒做一遍，让你真正理解：

- 为何用 `dyn_cast` 而不是乱 `cast`  
- 为何用 **Greedy Pattern** 而不是单次 walk  
- Pass 如何挂进 Pass Manager  

没有这步，后面读「图优化」会一直飘。

---

## 2. 目标效果

删除无副作用冗余（经典：`y = identity(x)`），所有用 `y` 的地方改用 `x`，并 erase 该 op；对测试用例可重复跑通。

---

## 3. 怎样做（步骤 + 每步为什么）

| 步骤 | 做什么 | 为什么 |
|---|---|---|
| 1 | 装 LLVM/MLIR 或能跑的 `mlir-opt` | 要白盒工具链 |
| 2 | 准备含 identity（或等价）的 `input.mlir` | 小输入可 diff |
| 3 | 写 `RewritePattern`：`dyn_cast`→`replaceAllUsesWith`→`eraseOp` | 标准安全改写 |
| 4 | 挂进 Greedy / canonicalize 管道 | 保证收敛、可组合 |
| 5 | `mlir-opt` 跑通并保存输出 | 可验收 |

伪代码见 [04](../04-mlir-pass-patterns.md)；C++ 设施见 [01](../01-cpp-foundations.md)。

---

## 4. 验收

- [ ] 至少一份输入被正确消除  
- [ ] 报告里写明：为何不用单次 walk；`cast`/`dyn_cast` 如何选  
- [ ] 边界：有副作用的 op **不能**删——你如何判断  

---

## 5. 和后面的连接

思想同构于 ATC「消除/融合」；面试常从这题拉开 Graph 优化话题。
