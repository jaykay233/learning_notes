# 03 · MLIR 基础：Dialect / Op / Type

> 原课：Dialect/Op/Type、语法、Builtin Dialect。  
> 昇腾推理日常用 ATC；本讲解决「Frontend 的 IR 到底长什么样」，并为 Lab1/[04](./04-mlir-pass-patterns.md) 打底。

---

## 0. 和上一讲怎么接

[02](./02-compiler-overview.md) 说 Frontend 要有 **High-level Graph IR**。  
MLIR 就是工业界把「多层 IR + 可扩展方言」做成 **可复用基础设施** 的一种答案。

```text
你在 MLIR 里看到的：
  Dialect = 谁家的算子包
  Op      = 一条具体运算
  Type    = 值的形状与元素类型
  Attribute = 编译期常量标注
```

---

## 1. 为什么是「Multi-Level」IR

### 1.1 问题

一步从「PyTorch 风格 Conv」降到「LLVM load/store」会怎样？

- 优化难写：融合该在哪一层做？  
- 硬件细节过早泄漏，换后端要推倒重来  
- 不同团队无法复用中间层（linalg/tosa 等）

### 1.2 怎样工作

渐进 lowering：

```text
高阶（接近框架：tosa.conv2d）
  → 中层（linalg / tensor 循环语义）
  → 低层（memref + 循环 / llvm）
  → 机器码或调运行时
```

每一层只解决一类问题；Pass 管道负责往下送（见 [04](./04-mlir-pass-patterns.md)）。

### 1.3 效果与用在哪

效果：优化可分层复用；硬件方言可插拔。  
用在哪：开源编译器、原课 tpu-mlir（Top→Tpu）、你自己的 Lab1。  
昇腾类比：Parser 后的 GE 图也是「中间层」，只是不一定以公开 `.mlir` 文本给你改。

---

## 2. Dialect（方言）

### 2.1 为什么要有 Dialect

若所有 Op 塞进一个全局扁平命名空间：

- 名字冲突（谁的 `add`？）  
- 无法按领域打包文档与 Pass  
- 无法「只链接我需要的方言」

Dialect = **带命名空间的 Op/Type/Attribute 包**。

### 2.2 怎样体现

文本里通常是 `方言名.op名`：

```mlir
%0 = arith.addi %a, %b : i32
%1 = tensor.empty() : tensor<2x3xf32>
```

常见方言角色：

| Dialect | 为什么存在 | 典型用在哪 |
|---|---|---|
| `builtin` | 模块、函数类型等「地面设施」 | 任何模块 |
| `func` | 函数边界 | `@main` |
| `arith` | 标量算术，便宜可折 | canonicalize |
| `tensor`/`tosa`/`linalg` | 张量计算中层 | 图优化与 lowering |
| 硬件方言 | 贴近某 NPU | 原课 TpuDialect |

原课 **TopDialect**：靠近框架的高阶图。  
原课 **TpuDialect**：靠近芯片。  
本课用 **GE 图** 类比二者之间的旅程，不要求你实现 TopDialect。

### 2.3 效果

换硬件 ≈ 换后端方言 + lowering，而不是重写前端。  
社区可共享 `tosa`/`linalg` 上的通用 Pass。

---

## 3. Operation（Op）

### 3.1 为什么「一切皆 Op」

控制流、函数调用、甚至 `module` 在 MLIR 里都统一成 Operation 模型，这样：

- 同一套 walk / pattern 能遍历  
- 同一套 verifier / 打印机制  

### 3.2 一个 Op 里有什么（实现级概念）

```text
Operation
  ├─ name（属于某 Dialect）
  ├─ operands  （输入 SSA 值）     ← 常用 SmallVector/ArrayRef 处理
  ├─ results   （输出 SSA 值）
  ├─ attributes（编译期常量字典）
  ├─ regions   （可选嵌套：循环体、函数体）
  └─ location  （调试源位置）
```

```mlir
%sum = arith.addi %a, %b {some_attr = 1 : i32} : i32
```

### 3.3 效果与用在哪

效果：Pass 用统一 API 改写图。  
用在哪：Lab1 的 `dyn_cast<YourOp>`、`replaceAllUsesWith`、`eraseOp`（[01](./01-cpp-foundations.md) RTTI）。

---

## 4. Type 与 Attribute

### 4.1 为什么类型要一等公民

张量编译的正确性大量来自 **shape/dtype**：错了不是小 bug，是整网烂掉。Infershape、verifier、codegen 都靠 Type。

### 4.2 怎样区分 Type vs Attribute

| | Type | Attribute |
|---|---|---|
| 描述 | 运行值的类型 | 编译期已知的标注 |
| 例子 | `tensor<1x3x224x224xf32>`、`i32` | `{value = dense<0> : ...}`、融合标记 |
| 何时变 | 随 lowering 可能变（tensor→memref） | 常在优化中被读写 |

### 4.3 效果与用在哪

- 前端导入：给每个 Value 建 Type  
- 图准备：Infershape 补全 `?`  
- Pattern：匹配「两输入同类型才融合」  

---

## 5. 文本 IR 与 Module 结构

### 5.1 为什么要有人类可读文本

- 单测：小 `.mlir` 文件当输入  
- 调试：`mlir-opt` 看每一跳  
- 教学：比纯 C++ API 直观  

### 5.2 典型结构

```mlir
module {
  func.func @main(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    // SSA: 每个 %x 只赋值一次
    %c = "tosa.conv2d"(%arg0, %w) : ...
    return %out : tensor<1x1000xf32>
  }
}
```

| 概念 | 含义 |
|---|---|
| `module` | 编译单元 |
| SSA `%name` | 单赋值，方便 dataflow 分析 |
| `func.func` | 函数（与 builtin 类型系统协作） |

工具：`mlir-opt`（跑 Pass）、`mlir-translate`（部分互转）。

### 5.3 效果与用在哪

Lab1：写 input.mlir → opt → diff output。  
面试：能当场读三行 IR 标出 Dialect/Op/Type。

---

## 6. Builtin 与「你至少该眼熟」的方言

**为什么单列 Builtin：** 没有它，module/函数类型无处安放——它是地基，不是「业务算子包」。

你应能认出：

| 类别 | 例子 | 何时出现 |
|---|---|---|
| 模块函数 | `module`、`func.func`、`return` | 永远 |
| 算术 | `arith.addi`、`arith.cmpf` | 标量/索引 |
| 张量 | `tensor.extract` / `insert` | 造型 |
| 控制流 | `scf.for` / `if` | lowering 后常见 |
| TOSA | `tosa.*` | 框架→中层中转 |

练习：打印一份 IR，逐行填「Dialect / Op / Type」。

---

## 7. 和昇腾 / 原课的对照（防错位）

| 概念 | 本课说法 |
|---|---|
| TopDialect | ≈ Parser 之后的高阶 GE 图 |
| TpuDialect | ≈ 硬件侧算子 + kernel |
| 你手写 Dialect | Lab1 可选；上板 **非必须** |
| lowering 链 | ATC：准备→拆分→优化→编译 |

---

## 8. 串到下一讲

有了 Dialect/Op/Type，才能谈：

- 用 **Pass** 批量改 Op  
- 用 **Pattern** 匹配局部 Dialect 形状  

→ [04](./04-mlir-pass-patterns.md)

---

## 9. 练习

1. 为什么不建议「一个大扁平 IR 打天下」？  
2. Op 的 operands 与 attributes 差别？  
3. `tensor<1x3xf32>` 是 Type 还是 Attribute？`{dilations = [1,1]}` 呢？  
4. 读三行陌生 `.mlir`，口头标注 Dialect/Op/Type。  
5. 用类比解释：TopDialect→TpuDialect 与 Parser→GE 后端。
