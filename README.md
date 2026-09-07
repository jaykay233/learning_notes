# learning_notes

AI 相关学习笔记，随读随记。

## 目录

```
models/
└── deepseek_v4/
    ├── architecture.md   # SGLang 中 DeepSeek-V4 整体结构
    ├── mhc.md            # mHC（流形约束超连接）详解与伪代码
    └── mqa.md            # MQA / Attention Layer 详解
```

## DeepSeek-V4

基于 [SGLang](https://github.com/sgl-project/sglang) 源码梳理：

| 文档 | 内容 |
|---|---|
| [architecture.md](models/deepseek_v4/architecture.md) | 模块树、MQA / CSA / HCA、MoE、KV 与压缩状态 |
| [mhc.md](models/deepseek_v4/mhc.md) | mHC 动机、公式、Sinkhorn、层内数据流与伪代码 |
| [mqa.md](models/deepseek_v4/mqa.md) | MQA 投影形状、单 KV head、RoPE/nope、CSA/HCA 与 Attention 数据流 |

## License

[MIT](LICENSE)
