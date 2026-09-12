# 08 · Frontend：模型导入与中间图（原 TopDialect / converter）

> 原课：TopDialect、从 onnx/torch/tflite converter。  
> 昇腾：**Framework Parser → GE 中间图**。

---

## 1. 为什么需要 Frontend

### 1.1 问题

PyTorch 的 aten op、TF 的 op、ONNX 的 op **名字与语义不完全相同**。  
若 Backend 直接面对三套，硬件映射要写三遍。

### 1.2 怎样工作

```text
框架文件
  → 解析：读节点、边、权重
  → 算子映射：Conv/Relu/… → 统一中间 Op
  → 初建 dtype/shape（后续 Infershape 再补）
  → 得到硬件无关（或弱相关）的中间图
```

原课把这层叫做 **TopDialect**；converter = 各框架 → Top 的适配器。

### 1.3 效果

后面所有融合/DCE/lowering 只认中间图。  
**用在哪：** ATC 一打开模型就先 Parser；失败也最先炸在这里。

---

## 2. Parser 细节（概念实现）

| 步骤 | 为什么做 | 失败时你看到什么 |
|---|---|---|
| 读模型 | 拿到图+权重 | 文件损坏/路径错 |
| 映射 op | 统一语义 | **unsupported op** |
| 建类型 | 后续优化依赖 | dtype 不合法 |
| 常量折叠入口 | 权重进图 | 巨大 Constant |

**ONNX 为何本课默认：** 跨框架导出成熟，课上少绑训练框架版本。

---

## 3. 和 MLIR Frontend 的对照

| tpu-mlir | 昇腾 | 你的动作 |
|---|---|---|
| converter→Top IR | Parser→GE 图 | 选 framework、修导出 |
| 可 `mlir-opt` 看 Top | 多靠日志/官方工具 | Lab3 跟踪日志 |
| Top 上前端 Pass | 图准备/原图优化 | 理解「还在前端」 |

---

## 4. 实践上你能控制什么

**为什么这些琐事重要：** 80%「转不过」是导出脏，不是 ATC 坏了。

- 固定 opset，避免花活自定义 op  
- `input_shape` 与预处理一致  
- 动态控制流过脏时：先静态化再导出  
- 不支持 op：改写子图 / 自定义算子 / 升 CANN 版本  

---

## 5. 练习

1. Frontend 的输出交给谁？  
2. unsupported op 应先查导出还是先怀疑 GE 后端？  
3. 用自己的话定义 TopDialect（即使昇腾不暴露它）。

### 参考答案

**1. Frontend 的输出交给谁？**

交给 **后端 / 图引擎**——昇腾里主要是 **GE**（再经优化、拆分、编译出 om）。

链路：

```text
框架模型（ONNX/…）
  → Frontend / Parser（统一中间图）
  → GE（准备 / 优化 / 内存与调度 / 编译）
  → *.om
```

Frontend **不**直接喂给 AscendCL：Deploy 吃的是 om；Parser 的产出是「后面所有 Pass 认的那张中间图」。

**2. unsupported op：先查导出，还是先怀疑 GE？**

**先查导出 / Parser 映射，不要先怀疑 GE 后端。**

原因：

- `unsupported op` 出现在 **算子映射** 阶段：框架里的某个节点名/语义，Parser 表里没有对应中间 Op  
- 这通常是：opset 太新、自定义 aten、导出花活、动态控制流没静态化——属于 **Frontend 边界**  
- GE 还没真正开始做硬件 lowering；后端「算子实现缺」类问题往往是另一类报错（编译/选型失败），时序也更靠后  

排错顺序：固定 opset → 看是哪个 ONNX/框架 op → 改写子图或升 CANN → 仍不过再查是否需自定义算子。课上说的「80% 转不过是导出脏」就是这个意思。

**3. 用自己的话定义 TopDialect**

**TopDialect =「框架无关、硬件弱相关」的顶层中间方言（中间图约定）。**

- **从哪来：** 各框架 converter/Parser 把 PyTorch/TF/ONNX 的异构 op **翻译**成同一套 Top 算子（Conv、Relu、…）  
- **干什么用：** 后面的融合、DCE、shape 推断、再 lowering 到硬件方言，都只认这一层，避免 Backend 为三套框架各写一遍  
- **昇腾对应物：** 不叫 TopDialect，但 **Parser → GE 中间图** 扮演同一角色——先统一语义，再进图优化与编译  

类比：Top ≈ 编译课里的「中端 IR」；converter/Parser ≈ 「把源语言解析进中端」的前端。

下一讲：[09](./09-backend-ge.md)
