# MLIR 实战系列

这套教程以 [KEKE046/mlir-tutorial](https://github.com/KEKE046/mlir-tutorial) 为主线，按“读 IR → 写 IR → 定义方言 → 写变换 → 做 lowering”的顺序重写。原仓库适合建立整体认识，但它最后更新于 2023 年，部分 C++ API 已经与当前 MLIR 不兼容。因此这里不以逐行复刻原仓库为目标，而是保留它的问题链和 Toy 示例，用可运行代码重建关键环节。

## 先确认版本

本教程当前按 **LLVM/MLIR 19.1.7** 编写，并在这台机器上实际验证。MLIR 的 C++ API、TableGen 接口和 Pass 注册方式变化很快；换 LLVM 版本时，优先运行示例，再调整代码，不要把别的版本的片段直接粘贴进来。

```bash
mlir-opt --version
llvm-config --version
```

原仓库 README 绑定的 LLVM 提交是 `186a4b3b657878ae2aea23caf684b6e103901162`。如果要复现 2023 年的原版实验，应使用那个提交；如果要跟本教程学习，请以本目录的代码和 19.1.7 的输出为准。

## 课程路线

```text
文本 IR
  → Operation / Block / Region 树
  → SSA、def-use 与遍历
  → MLIRContext / Parser / OpBuilder
  → TableGen 定义 Toy Dialect
  → Op 约束、verifier、assembly format
  → Pass 与 DCE
  → Pattern Rewrite 与 CSE
  → TypeConverter 与 Dialect Conversion
  → 把 Toy 操作降低到 arith/LLVM
```

| 课 | 主题 | 对应原仓库 | 学完能做什么 | 状态 |
|---|---|---|---|---|
| [01](./01-operation-tree.md) | Operation / Block / Region | README 2.1 | 读懂嵌套 IR，区分语法糖和 generic form | 已开放 |
| 02 | SSA、def-use、walk | README 2.3、3、5 | 追踪一个 Value 的定义和所有使用，读懂 CSE 为什么安全 | 下一篇 |
| 03 | 读入、打印、构造 IR | `ex1-io`、`ex2-build` | 写 C++ 工具解析 `.mlir`，用 `OpBuilder` 生成函数 | 待开放 |
| 04 | 最小 TableGen Dialect | `ex3-dialect` | 让 `toy-opt` 认识 `toy.add` 和 `toy.const` | 待开放 |
| 05 | Op 约束与 verifier | `ex4-beautiful-dialect` | 让错误 IR 在 parse/verify 阶段失败，而不是等到后端崩溃 | 待开放 |
| 06 | 自定义 assembly format 与函数接口 | `ex4-beautiful-dialect` | 写出可读的 `toy.func`、`toy.call`、`toy.ret` | 待开放 |
| 07 | Pass 与死代码消除 | `ex5-pass` | 注册 `toy-dce`，观察无用 Op 被删除 | 待开放 |
| 08 | Pattern Rewrite 与 CSE | `ex6-pattern` | 匹配 `toy.add(x, x)`，理解 rewrite 的不动点与迭代上限 | 待开放 |
| 09 | 自定义 Type 与类型转换 | `ex7-convert` | 把 `!toy.int<32>` 转成 `i32` | 待开放 |
| 10 | Toy 到 arith 的完整 lowering | `ex7-convert` | 串起一条最小、可验证的 lowering 管线 | 待开放 |
| 11 | 综合项目 | 全仓库 | 给 Toy 增加一种 Op，补 verifier、rewrite 和 lit 风格测试 | 待开放 |

## 学习方式

每课只解决一个可以独立验证的问题，顺序不建议跳。每篇都包含：

1. 本课在整条编译链中的位置；
2. 先给心智模型，再出现 API 名称；
3. 完整可运行输入、命令和预期输出；
4. 一行或一个阶段的具体执行轨迹；
5. 常见错误的症状；
6. 自检题和答案。

建议每次只学一课。先自己预测输出，再运行命令核对；如果预测错了，记录错在“语法层、IR 结构层还是 API 层”。这比连续读十篇更有用。

## 当前环境

```bash
mlir-opt --version
# LLVM (http://llvm.org/):
#   LLVM version 19.1.7

llvm-config --version
# 19.1.7
```

第 01 课只需要 `mlir-opt`。从第 03 课开始需要 MLIR 的 CMake 配置和静态库；当前机器上的路径是：

```bash
MLIR_DIR="$(llvm-config --prefix)/lib/cmake/mlir"
LLVM_DIR="$(llvm-config --prefix)/lib/cmake/llvm"
```

macOS 上使用 conda 的 MLIR 包构建时，CMake 项目要同时启用 C 和 CXX：

```cmake
project(my-mlir-tool LANGUAGES C CXX)
```

## 参考边界

原仓库提供的是一份 2023 年的实践路线，而不是当前 API 手册。本系列会明确标出：

- 哪些是稳定的 MLIR 概念，例如 Operation、Block、Region、SSA、Pattern；
- 哪些是随版本变化的 C++/TableGen 写法；
- 哪些结论已经通过本机运行验证，哪些只是静态阅读。

遇到 API 冲突时，以当前版本头文件和运行结果为准，不把旧教程中的“能编译”当成事实。

## 下一篇

[01 · Operation、Block 与 Region](./01-operation-tree.md)
