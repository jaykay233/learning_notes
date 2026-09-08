# 14 · YOLOv8 目标检测昇腾部署

> 原课：letterbox 预处理、NMS 后处理。

---

## 0. 为什么检测模型要拆三截

```text
预处理（letterbox）→ 网络推理（om）→ 后处理（decode+NMS）
```

网络只认固定数值张量；业务要的是「图上的框」。  
中间两段常放 CPU，中间一段放 NPU——边界清晰才能流水（Lab5）。

---

## 1. Letterbox——为什么与怎样

### 1.1 为什么

直接 resize 到正方形会 **拉变形**，框映射回原图会错。  
Letterbox：等比缩放 + 填充灰边，保持长宽比。

### 1.2 怎样

```text
算 scale = min(目标W/原W, 目标H/原H)
缩放 → 居中 pad
记录 scale 与 pad，后处理把框乘回去
```

### 1.3 效果与坑

效果：几何正确。  
坑：pad 值、色序、均值方差必须与训练一致；否则「NPU 数值对但框飞」。

---

## 2. 推理 om——为什么常固定 shape

AOT 友好：固定 `1x3xHxW`。  
动态分辨率：要在 ATC 侧开动态档（概念见 [02](./02-compiler-overview.md) AOT），成本更高。

---

## 3. NMS——为什么常放 CPU

### 3.1 为什么

- 控制流、变长框列表，NPU 图支持未必划算  
- 业务要调阈值，CPU 改参数更灵活  

若后处理也进 ONNX：要确认算子支持与性能。

### 3.2 怎样（概念）

按类别排序分数 → 算 IoU → 抑制重叠框。

---

## 4. 端到端数据流

```text
原图 → letterbox+normalize → Device 输入
     → AscendCL Execute
     → 输出张量回 Host
     → decode → NMS → 画框
```

---

## 5. 练习

1. 为何必须保存 letterbox 的 scale/pad？  
2. 后处理进图 vs 放 CPU 的利弊？  
3. 预处理错了，你会误判成「量化精度问题」吗？如何避免？

下一讲：[Lab5](./labs/lab5-yolo-stream.md)
