# Lab2 · ResNet50：CPU vs 昇腾基础推理

> 原课 Lab2。前置：[06](../06-environment-setup.md)、[07](../07-toolchain-workflow.md)。

---

## 1. 为什么做这个 Lab

第一次把 [02](../02-compiler-overview.md) 的地图 **落到真机器**：

```text
ONNX → ATC(AOT) → om → 设备执行
```

并建立「CPU 基线 vs NPU」的对比习惯——后面所有性能/精度实验都靠这个习惯。

---

## 2. 目标效果

- 成功产出 `resnet50.om`  
- 设备上跑通前向  
- 表格记录延迟；Top-1（或logits）与 CPU 对齐（允许微小数值差）

---

## 3. 怎样做（步骤 + 为什么）

| 步骤 | 做什么 | 为什么 |
|---|---|---|
| 1 | 准备固定 shape 的 ONNX | AOT 需要可推断形状 |
| 2 | `atc` 指定 `soc_version`、input_shape | 绑硬件、绑输入 |
| 3 | 造与训练一致的 bin 输入 | 避免「预处理错当编译错」 |
| 4 | msame/AscendCL 执行 | Deploy 最小闭环 |
| 5 | CPU ORT/PyTorch 对比 | 正确性门禁 |

命令以当前 CANN 文档为准。

---

## 4. 验收

- [ ] om + 一次成功推理  
- [ ] CPU/NPU 延迟表  
- [ ] 书面：ATC 与 AscendCL 各在 Transform/Deploy 哪一侧  

---

## 5. 连接

下一 Lab3 在同一模型上 **跟踪日志**；量化 Lab4 以「FP 已正确」为前提。
