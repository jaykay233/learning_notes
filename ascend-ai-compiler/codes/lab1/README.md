# Lab1 实验材料 · 冗余算子消除

对应讲义：[../../ascend-ai-compiler/labs/lab1-redundant-op-pass.md](../../ascend-ai-compiler/labs/lab1-redundant-op-pass.md)

本目录先备齐 **输入 IR、期望输出、方言/Pass 骨架**。你装好 LLVM/MLIR 后，按 `README` 补全 `matchAndRewrite` 即可验收。

## 目录

```text
codes/lab1/
├── README.md                 # 本文件
├── inputs/                   # 测试输入（含 lab.identity）
├── expected/                 # Pass 跑完后的期望 IR（语义级）
├── include/lab1/             # C++ 头文件骨架
├── lib/                      # Dialect + Pass 骨架
├── tools/lab1-opt/           # 迷你 mlir-opt（注册 lab 方言与 Pass）
├── test/lit.local.cfg.example
└── scripts/check_diff.sh     # 有 lab1-opt 后做期望对比
```

## 目标

消除无副作用的：

```mlir
%y = lab.identity %x : tensor<4xf32>
```

所有使用 `%y` 的地方改为 `%x`，并删掉该 op。

## 输入用例说明

| 文件 | 测什么 |
|---|---|
| `inputs/01_simple_identity.mlir` | 单个 identity，结果直接 return |
| `inputs/02_chain_identity.mlir` | 连续两个 identity（测 Greedy 收敛） |
| `inputs/03_identity_then_use.mlir` | identity 后再做 arith，测 replaceAllUses |
| `inputs/04_no_identity.mlir` | 负例：没有 identity，图应保持不变 |
| `inputs/05_side_effect_note.mlir` | **文档用例**：说明有副作用的 op 不能当 identity 删（见文件内注释） |

`expected/` 下同名文件是消除后的参考形态（允许 SSA 名字不同，看数据流等价即可）。

## 依赖

本机已可用（conda-forge，无需 sudo）：

```bash
# 新开终端后应直接可用；或先：
export PATH="$HOME/.local/opt/miniconda3/bin:$PATH"
mlir-opt --version   # LLVM 19.x（conda-forge）
```

- `mlir-opt` / `mlir-tblgen`：`$HOME/.local/opt/miniconda3/bin/`
- 编 Lab1 时：`MLIR_DIR=$HOME/.local/opt/miniconda3/lib/cmake/mlir`
- macOS 构建需：`-DLLVM_ENABLE_LIBCXX=ON`，且 `CMakeLists.txt` 启用 `C`（LLVM 选项探测需要）

> 本机当前是 **LLVM/MLIR 19**。若你换了版本，头文件/库名可能差一截，按本机头文件微调即可。

## 构建（本机已验证）

```bash
export PATH="$HOME/.local/opt/miniconda3/bin:$PATH"
export MLIR_DIR="$HOME/.local/opt/miniconda3/lib/cmake/mlir"

cmake -G Ninja -S . -B build \
  -DMLIR_DIR="$MLIR_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_LIBCXX=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_C_COMPILER=/usr/bin/clang
cmake --build build
```

跑单个用例：

```bash
./build/bin/lab1-opt inputs/01_simple_identity.mlir \
  --pass-pipeline='builtin.module(func.func(lab1-eliminate-identity))'
```

批量对比（需已生成 `lab1-opt`）：

```bash
chmod +x scripts/check_diff.sh
./scripts/check_diff.sh ./build/bin/lab1-opt
```

## 填写标注（作业核心）

唯一必填逻辑在 `lib/EliminateIdentityPass.cpp` 的 `matchAndRewrite`，用注释框标出：

```text
// ========== BEGIN: Lab1 填写（学生作业核心）==========
...
// ========== END: Lab1 填写 ==========
```

当前填写内容（语义：用 input 替换 identity 的 result，并删除该 op）：

```cpp
rewriter.replaceOp(op, op.getInput());
return success();
```

其余文件是骨架（方言解析打印、`lab1-opt` 注册、CMake）。为适配 MLIR 19，骨架里有两处小改动（**不是**作业题干要求的填写点）：

| 文件 | 改动 | 原因 |
|---|---|---|
| `CMakeLists.txt` | `LANGUAGES C CXX` | LLVM CMake 探测需要 C |
| `lib/LabDialect.cpp` | `getOperation()->getResult(0)` | `OneResult` 不再提供无参 `getResult()` |

可选后续：把 Pass 挂进更长管道 `canonicalize, cse, lab1-eliminate-identity`；写短报告（为何 Greedy；`dyn_cast` vs `cast`；副作用边界）。

## 与讲义的对应

| 材料 | 讲义 |
|---|---|
| `lab.identity` | [01](../../ascend-ai-compiler/01-cpp-foundations.md) / Lab1「消 identity」 |
| Pattern + Greedy | [04](../../ascend-ai-compiler/04-mlir-pass-patterns.md) |
| `dyn_cast` | [01](../../ascend-ai-compiler/01-cpp-foundations.md) RTTI |
