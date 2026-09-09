# 04 · MLIR Pass 管理与模式匹配

> 原课：Pass Manager、GreedyPatternRewrite。  
> 前置：[01](./01-cpp-foundations.md)、[03](./03-mlir-dialect-op-type.md)。实践：[Lab1](./labs/lab1-redundant-op-pass.md)。  
> 缩写对照：[glossary.md](./glossary.md)。

---

## 0. 为什么有了 IR 还要 Pass

IR（Intermediate Representation，中间表示）只是「照片」；**Pass**（编译器的一遍变换/分析）才是「修图滤镜」。  
Frontend 里的融合、DCE（Dead Code Elimination，死码消除）、canonicalize（规范化），后端 lowering（降级），全是一条条 Pass（或等价黑盒步骤）串起来。

```text
mlir-opt 输入.mlir -pass-pipeline='...'
        ≈
ATC（昇腾张量编译器）/ GE（图引擎）内部一长串你看不见名字的图优化
```

本讲把白盒版讲清楚，你才能理解昇腾黑盒在干什么。

---

## 1. Pass 是什么

### 1.1 为什么要「插件化」变换

若把所有优化写进一个巨型函数：

- 无法单独开关「只跑融合」  
- 无法复用社区 CSE（公共子表达式消除）
- 测试极难  

→ 每个变换做成 Pass，由 **Pass Manager（PM）** 编排。

### 1.2 怎样工作

```text
Pass：对某一层级（Module / Func / Op）的一次分析或改写

PM：
  Module 级管道
    └─ 对每个 Func 嵌套跑：
         PassA → PassB → PassC
```

| 能力 | 含义 |
|---|---|
| 嵌套管道 | 外层 Module，内层逐 Func |
| Analysis | 只读建分析结果，供后续查询 |
| 失败 | `signalPassFailure`，可中止管道 |
| 注册 | TableGen / `PassWrapper`（CRTP，见 [01](./01-cpp-foundations.md)） |

### 1.3 效果与用在哪

效果：优化可组合、可测试、可命令行裁剪。  
用在哪：`mlir-opt`；你自己的 Lab1；思想上对应 ATC 优化阶段。

---

## 2. Pass Manager 为何值得单独学

### 2.1 为什么「顺序」是大问题

错误顺序举例：

- 先 lowering 到低层循环，再想做高层 Conv 融合 → **融合机会已没**  
- 先暴力改写再 canonicalize → Pattern 组合爆炸  

PM 强制你思考 **依赖与层级**。

### 2.2 怎样用（概念）

```bash
# 示意
mlir-opt input.mlir \
  --pass-pipeline='builtin.module(func.func(canonicalize,cse,my-eliminate-identity))'
```

读 pipeline：在 module 下，对每个 func 依次 canonicalize → cse → 你的 Pass。

### 2.3 效果

- 同一套 Pass 库，不同产品线拼不同管道  
- 出 bug 时可二分「关掉哪一段」  

### 2.4 用在哪

Lab1 把消除 Pass 插进 canonicalize 前后对比；昇腾侧用日志观察「优化是否发生」，对应「管道里有没有这类 Pass」。

---

## 3. Pattern Rewrite：为什么不只写 walk

### 3.1 walk 的局限

```cpp
op->walk([](Operation *op) {
  if (auto id = dyn_cast<IdentityOp>(op)) {
    // erase...
  }
});
```

可以工作，但：

- 消除 A 之后可能 **新暴露** 可消的 B；单次 walk 不一定收敛  
- 多个独立规则难组合、难复用  
- 与官方 canonicalize 生态不一致  

### 3.2 Pattern 怎样工作

每个规则实现：

```cpp
LogicalResult matchAndRewrite(IdentityOp op,
                              PatternRewriter &rewriter) const override {
  rewriter.replaceAllUsesWith(op.getResult(), op.getInput());
  rewriter.eraseOp(op);
  return success();
}
```

**GreedyPatternRewriteDriver：**

```text
把 Pattern 集合丢进驱动
  → 找匹配 → 改写 → 再找
  → 直到无匹配或触及迭代/复杂度上限
```

**为什么要有上限：** 坏 Pattern 可能来回抖；防止编译器死循环。

### 3.3 效果与用在哪

| | 单次 walk | Greedy Patterns |
|---|---|---|
| 收敛 | 不保证 | 在上限内尽量收敛 |
| 组合 | 差 | 多 Pattern 可叠加 |
| 典型 | 一次性统计 | canonicalize、消冗余、常量折 |

Lab1 = 标准 Greedy Pattern 练习。

---

## 4. Canonicalize vs 你的业务 Pass

### 4.1 为什么要分两层

- **Canonicalize：** 把 IR 收到少数「规范形」（方便后面匹配）  
- **业务 Pass：** 实现产品语义（融某两家 op、消 identity）  

若跳过规范形，你要为每一种歪写法则写 Pattern → 组合爆炸。

### 4.2 怎样配合

```text
建议管道：
  canonicalize / CSE
    → 你的消除 / 融合
    → 再 canonicalize（清理改写残渣）
```

许多 Op 自带 `fold` / `getCanonicalizationPatterns`（常由 TableGen/手写提供）。

### 4.3 效果与用在哪

效果：Pattern 更短、更稳。  
用在哪：所有认真的 MLIR 优化管道；面试常问。

---

## 5. 和 Visitor、RTTI、昇腾的连接

```text
walk / PM          ← 遍历与调度（Visitor 思想）
dyn_cast           ← 匹配具体 Op（[01] RTTI）
SmallVector        ← 暂存要改的 operands
Pattern + Greedy   ← 声明式局部改写
        │
        ▼ 思想同构
ATC/GE 里的融合、消除、常量折（黑盒）
```

---

## 6. 端到端：Lab1 在本讲中的位置

见 [labs/lab1-redundant-op-pass.md](./labs/lab1-redundant-op-pass.md)：  
你要交出的不是「会调用 mlir-opt」，而是理解 **为何用 Pattern+Greedy，以及如何挂进 PM**。

---

## 7. 练习

1. 为什么融合 Pass 通常放在 lowering 到 memref 循环 **之前**？  
2. Greedy 没有迭代上限会怎样？  
3. 何时用 walk 就够、何时必须 Pattern？  
4. 画一条 pipeline，标出 canonicalize 与你的消除 Pass 的位置及原因。  
5. 用一段话对比：`mlir-opt` 管道 vs ATC「图优化阶段」。

下一站：先做完 Lab1，再进入 [05](./05-ascend-npu-hardware.md)。
