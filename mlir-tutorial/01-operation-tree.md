# 01 · Operation、Block 与 Region

```text
本次讲解位置
主题：mlir-tutorial
章节：MLIR 基本结构
知识点：Operation、Block、Region 的嵌套关系
上次：无，作为系列起点
下次：SSA、def-use 链与 Op 遍历
验证环境：LLVM/MLIR 19.1.7
```

## 为什么现在讲这个

后面写 Dialect 和 Pass 时，C++ 代码里会不断出现 `Operation *`、`Block *`、`Region *`。如果只把这些词当成三个 API，很容易写出“能打印 IR，却不知道修改应该落在哪一层”的代码。

这一课要解决一个具体判断：给定一段 MLIR，某个 Op 位于哪个 Block，哪个 Region 拥有它，谁又是这个 Region 的终止符。这个判断决定 Pass 应该从哪个层级开始遍历，也决定删除、替换或插入一个 Op 时修的是哪棵树。

## 先建立心智模型

MLIR 的 IR 不是一张平铺的指令表，而是带递归结构的树：

```text
Operation
  ├─ operands：它读取的 SSA Value
  ├─ results：它产生的 SSA Value
  ├─ attributes：编译期已知的配置
  └─ regions
       └─ Block
            ├─ block arguments
            └─ Operation...
                 └─ 还可能再拥有 Region
```

三种对象各管一件事：

| 对象 | 回答的问题 | 例子 |
|---|---|---|
| Operation | 执行什么操作，读写哪些 SSA Value，带什么属性 | `func.func`、`scf.if`、`arith.addi` |
| Region | 某个 Op 内部的一块独立控制流区域 | `func.func` 的函数体、`scf.if` 的 then/else 体 |
| Block | Region 内的一组连续 Op，有入口参数，通常以终止符结束 | 函数入口块、`^bb0` |

关键区别是：**Region 不是 Block，Block 也不是 Op**。Function Op 拥有一个 Region；Region 由 Block 组成；Block 里放 Operation。一个 Op 可以拥有多个 Region，例如 `scf.if` 的 then 和 else。

## 完整可运行示例

文件：[`examples/01-operation-tree.mlir`](./examples/01-operation-tree.mlir)

```mlir
module {
  func.func @choose(%cond: i1, %a: i32, %b: i32) -> i32 {
    %0 = scf.if %cond -> i32 {
      scf.yield %a : i32
    } else {
      scf.yield %b : i32
    }
    return %0 : i32
  }
}
```

运行：

```bash
cd /Users/saboxu/Downloads/learning_notes/mlir-tutorial
mlir-opt examples/01-operation-tree.mlir
```

预期输出与输入语义相同。`mlir-opt` 会重新打印规范化后的 SSA 名称：

```mlir
module {
  func.func @choose(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
    %0 = scf.if %arg0 -> i32 {
      scf.yield %arg1 : i32
    } else {
      scf.yield %arg2 : i32
    }
    return %0 : i32
  }
}
```

这里输入中的 `%cond`、`%a`、`%b` 被打印成 `%arg0`、`%arg1`、`%arg2`，不是语义变化。MLIR 文本中的 SSA 名称只是显示名称，真实关系由 Value 的身份决定。

## 把语法糖展开成通用形式

`func.func`、`scf.if` 等 Op 定义了自己的 assembly format，所以平时看到的是便于阅读的缩写。加 `--mlir-print-op-generic` 可以强制输出通用形式：

```bash
mlir-opt --mlir-print-op-generic examples/01-operation-tree.mlir
```

完整预期输出见 [`examples/01-operation-tree.generic.mlir`](./examples/01-operation-tree.generic.mlir)。核心片段是：

```mlir
"builtin.module"() ({
  "func.func"() <{function_type = (i1, i32, i32) -> i32, sym_name = "choose"}> ({
  ^bb0(%arg0: i1, %arg1: i32, %arg2: i32):
    %0 = "scf.if"(%arg0) ({
      "scf.yield"(%arg1) : (i32) -> ()
    }, {
      "scf.yield"(%arg2) : (i32) -> ()
    }) : (i1) -> i32
    "func.return"(%0) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
```

通用形式不适合日常手写，但它把两个事实暴露出来：

1. `func.func` 后面的 `{ ... }` 是一个 Region，而 Region 的第一个 Block 被打印成 `^bb0`。
2. `scf.if` 后面的两对 `{ ... }` 是两个 Region，分别对应 then 和 else。它们不是两个普通属性，也不是一条 Op 指令的两个分支列表。

## 按层走一遍

以上面这段 IR 为例，从最外层开始拆：

| 层 | 对象 | 具体内容 |
|---|---|---|
| 1 | `builtin.module` Operation | 拥有一个 Region，Region 内包含模块入口 Block |
| 2 | `func.func @choose` Operation | 位于模块 Region 的 Block 中，拥有函数体 Region |
| 3 | `^bb0(%arg0, %arg1, %arg2)` | 函数体 Region 的入口 Block，携带三个参数 |
| 4 | `scf.if` Operation | 位于 `^bb0`，读取 `%arg0`，产生 `%0`，拥有两个 Region |
| 5 | then Region | 包含 `scf.yield %arg1` |
| 6 | else Region | 包含 `scf.yield %arg2` |
| 7 | `func.return` | 位于 `^bb0`，读取 `%0`，结束函数返回路径 |

`scf.yield` 只结束它所在的 SCF Region 并交回结果；`func.return` 才结束整个函数。两者都叫“返回”会让初学者误判作用域，但它们的父 Op 不同。

## Operand、Result 与 Attribute 怎么分

看 `scf.if` 这一行：

```mlir
%0 = scf.if %cond -> i32 {
  ...
}
```

- `%cond` 是 operand，运行期要读取的 SSA Value。
- `%0` 是 result，由 `scf.if` 产生，类型为 `i32`。
- `-> i32` 声明结果类型，不是额外创建了一个值。
- 如果 Op 上写着 `{some_flag = true}`，那通常是 attribute，属于编译期配置。

再对比函数：

```mlir
func.func @choose(%cond: i1, %a: i32, %b: i32) -> i32
```

`@choose` 是符号名，函数签名中的类型属于函数类型信息；函数体是 Region。不要因为 `func.func` 和 `scf.if` 都能包含 `{ ... }`，就误以为它们位于同一层。一个是模块里的函数定义，一个是函数体 Block 里的控制流 Op。

## 一个常见错误及其症状

错误理解：

```text
%0 是变量，scf.if 执行一次就给它赋值；
then 和 else 是两个基本块，所以都属于当前函数 Block。
```

由此会得到一个错误动作：想统计函数里的分支数时，只遍历当前 Block 的 Op；或者想替换 `%0` 时，直接改 `scf.yield` 的操作数。

实际结构是：`scf.if` 是当前 Block 中的一个 Op；then/else 是它拥有的两个 Region，每个 Region 自己包含 Block。正确做法是先定位 `scf.if`，再分别进入它的 Region。若只做 Block 级遍历，会看到 `scf.if`，但不会自动把两个 Region 里的 Op 当成当前 Block 的直接成员。

症状通常在两类任务中出现：

- 遍历统计时少算了嵌套 Region 中的 Op；
- 重写时在错误的 Block 插入 Op，验证器报“definition does not dominate use”或 Region 终止符不合法。

## 验证边界

本课只验证文本 IR 的解析、打印和结构展开，没有编写 C++ Pass。已经实际运行：

```bash
mlir-opt --version
mlir-opt examples/01-operation-tree.mlir
mlir-opt --mlir-print-op-generic examples/01-operation-tree.mlir
```

下一步会把这种树结构映射到 C++ API：`Operation::getRegions()`、`Region::getBlocks()`、`Block::getOperations()`，以及 `walk` 如何递归进入嵌套 Op。

## 自检

### 1. `func.func` 拥有的是 Block 还是 Region？

它拥有一个 Region；该 Region 默认包含一个入口 Block。函数参数对应的 block arguments 在入口 Block 上。

### 2. `scf.if` 的 then 和 else 分别是什么？

两个 Region。每个 Region 里至少有一个 Block；示例中各自的 Block 只包含一条 `scf.yield`。

### 3. `%0` 是 operand、result 还是 attribute？

它是 `scf.if` 的 result。`%cond` 是 operand；`-> i32` 声明 result 的类型。

### 4. 为什么 `scf.yield` 不等于函数返回？

它只向父级 `scf.if` Region 交回值。函数返回由当前函数体中合适的终止 Op 完成，本示例是 `func.return`。

### 5. generic form 对我们理解 IR 有什么帮助？

它去掉自定义 assembly format 的缩写，直接显示 Op 名称、operand、result、attribute 和 Region。遇到不熟悉的语法时，先打印 generic form，通常比猜自定义格式更快。
