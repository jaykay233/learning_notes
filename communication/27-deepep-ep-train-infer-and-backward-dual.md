# DeepEP：推理/训练走 EP，反传是 dispatch↔combine 对偶

> 承接 [26](./26-deepep-v2-pp-send-recv-ring.md)。走完 Elastic 旁路（Engram / PP）后，两个使用层问题要钉死：**(1) 推理还用不用 PP？(2) 训练梯度反传有没有单独代码？** 答案都落在官方 README 的产品定位与 training 示例上，而不是再开一套 grad kernel。

```text
本次讲解位置
章节：communication / DeepEP 使用层
小节：EP vs PP；dispatch/combine backward 对偶
知识点：主路径=EP；PP/Engram/CP=Experimental；反传=互换原语+复用 handle
上次：PP 环 send/recv（26）
下次：AGRS session（若继续 Elastic）或回到 EP 某条细路径
原语：ElasticBuffer.dispatch / combine；EPHandle；无内置 autograd.Function
```

## 为什么现在讲这个

kernel 走读容易默认「库里有的 API 推理都会用」。README 写得很清楚：DeepEP **聚焦 EP**；PP / Engram / CP 是实验旁路。
另一侧，训练反传**没有** `grad_dispatch.cuh`，也没有仓库内 `torch.autograd.Function`——误以为「没代码就不能训」或「反传另开 RDMA 路径」都会错。本文把产品边界和反传对偶写成可检索的一页。

---

## 1. 产品边界：主路径 vs 实验旁路

```3:3:README.md
DeepEP ... currently focuses on expert parallelism (EP) ... while also offering
experimental primitives for pipeline parallelism (PP), context parallelism (CP),
and remote memory access (Engram) ...
```

```31:31:README.md
- Engram, PP, and CP are experimental features
```

| 能力 | 定位 | 典型用途 |
|---|---|---|
| **EP** `dispatch` / `combine` | **主推** | MoE 训练、推理 prefilling、推理 decoding |
| **PP** `pp_send` / `pp_recv` | Experimental，「0 SM PP」 | 流水线环邻接传 tensor（训练 PP 最常见） |
| **Engram** | Experimental | 按 index 远程 GET |
| **CP** | Experimental | Copy Engine 相关 |

库整体标称 *training and inference*，但 **推理示例接的是 EP**，不是 PP。

```text
推理 / 训练 MoE 数据面  →  ElasticBuffer.dispatch / combine
流水线 stage 传激活    →  pp_send / pp_recv（可选实验，非默认）
```

---

## 2. 推理还用 PP 吗？

**不用 PP 当默认路径。** 要用 DeepEP 做 MoE 推理，走 README 的：

- *Example use in model training or inference prefilling* → `dispatch` / `combine`
- *Example use in inference decoding* → 同一 `ElasticBuffer`，可缓存 `handle`

PP 语义是经典 Pipeline Parallel（prev/next），和 expert all-to-all 正交。仓库内 PP 只有 `tests/elastic/test_pp.py` 与实验 API，**没有**挂进 decoding 示例。

| 问题 | 答案 |
|---|---|
| DeepEP 推理主通信？ | EP |
| PP 能否理论上服务「层切多卡流水线推理」？ | 原语可以，但非本库主推集成 |
| 读 `pp_send_recv.cuh` 是为了理解推理 MoE？ | 否；那是旁路 |

---

## 3. 训练梯度怎么反传？有代码吗？

### 3.1 结论

- **有用法代码**：README（及 `docs/legacy.md`）里的 `dispatch_backward` / `combine_backward`。
- **没有**单独 grad kernel，**没有** DeepEP 内置 `autograd.Function`。
- 反传 = **同一套** `combine` / `dispatch`，方向对偶，**复用前向 `EPHandle`**。

### 3.2 对偶表

| 前向 | 数学直觉 | 反传调用 |
|---|---|---|
| `dispatch`：token → expert | 按路由散射 | `combine`：expert 上 grad → 源 rank |
| `combine`：expert 结果 → 源 | 按路由归约 | `dispatch`：源上 grad → 各 expert |

```207:221:README.md
def dispatch_backward(grad_recv_x, grad_recv_topk_weights, handle):
    """The backward pass of MoE dispatch is actually a combine."""
    combined_grad_x, combined_grad_topk_weights, event = _buffer.combine(
        grad_recv_x, handle=handle, topk_weights=grad_recv_topk_weights, ...)
```

```239:252:README.md
def combine_backward(grad_combined_x, handle):
    """The backward pass of MoE combine is actually a dispatch."""
    grad_x, _, _, _, event = _buffer.dispatch(
        grad_combined_x, handle=handle, ...)
```

```text
Forward:
  x  --dispatch(handle)-->  recv_x  --expert GEMM-->  y_e  --combine(handle)-->  y

Backward:
  dL/dy  --dispatch(handle)-->  dL/dy_e  --GEMM^T-->  dL/drecv_x  --combine(handle)-->  dL/dx
           (= combine_backward)                         (= dispatch_backward)
```

`handle` 前向算好路由后缓存；反传**不再重算** topk 布局。

### 3.3 谁接 autograd？

DeepEP 只给通信原语。把上述函数挂进：

```python
class MoEDispatch(torch.autograd.Function):
    @staticmethod
    def forward(...): ...
    @staticmethod
    def backward(...):  # 内部调 buffer.combine
        ...
```

是 **Megatron / 自研 MoE 层** 的事。在 `deep_ep/` 下 grep `autograd` 不会找到现成包装类——这是刻意分层，不是缺反传能力。

### 3.4 与 PP / Engram 无关

| 路径 | 是否承担 MoE 训练反传 |
|---|---|
| EP `dispatch`/`combine` | **是**（对偶） |
| PP | 否（那是 pipeline stage 通信） |
| Engram | 否（远程 KV-like GET） |

---

## 4. 可运行小验证：对偶映射表

文件：`communication/27_ep_backward_dual_sim.py`（纯逻辑，不跑 GPU）。

```python
#!/usr/bin/env python3
"""Map MoE EP forward ops to their DeepEP backward duals."""

from __future__ import annotations


DUAL = {
    "dispatch": "combine",  # dispatch_backward
    "combine": "dispatch",  # combine_backward
}


def backward_comm(forward_op: str) -> str:
    return DUAL[forward_op]


def main() -> None:
    print("EP training backward dual (same kernels, reuse handle):")
    for fwd in ("dispatch", "combine"):
        print(f"  forward {fwd:8s} -> backward calls {backward_comm(fwd)}")

    print("product roles:")
    print("  inference MoE  -> EP (not PP)")
    print("  training MoE   -> EP forward + dual backward")
    print("  PP/Engram/CP   -> experimental side paths")


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 communication/27_ep_backward_dual_sim.py
```

期望：

```text
EP training backward dual (same kernels, reuse handle):
  forward dispatch -> backward calls combine
  forward combine  -> backward calls dispatch
product roles:
  inference MoE  -> EP (not PP)
  training MoE   -> EP forward + dual backward
  PP/Engram/CP   -> experimental side paths
```

完整可抄示例：上游 DeepEP 仓库 `README.md`「Example use in model training or inference prefilling」；Legacy：`docs/legacy.md` 同名对偶。

---

## 5. 常见误判

| 误判 | 实际 |
|---|---|
| 推理默认走 PP | 默认走 EP；PP 是实验旁路 |
| 训练反传另有 grad RDMA kernel | 就是 `combine`/`dispatch` 对偶 |
| 仓库没有 backward 就不能训 | README 已给用法；缺的是框架层 autograd 包装 |
| 反传要重新 layout | 复用前向 `EPHandle` |
| Engram fetch 参与 MoE backward | 不参与 |

---

## 6. 自测题

1. **DeepEP 推理 MoE 主 API 是什么？**
   答：`ElasticBuffer.dispatch` / `combine`。

2. **`dispatch` 的 backward 调什么？**
   答：`combine`（同一 `handle`）。

3. **`combine` 的 backward 调什么？**
   答：`dispatch`。

4. **库内有没有现成 `torch.autograd.Function`？**
   答：没有；由上层训练框架包装。

5. **PP 和 MoE 梯度反传是一条路径吗？**
   答：不是。

---

## 7. 学习进度

- [x] PP 环机制（26）
- [x] EP 训练/推理边界 + 反传对偶（本篇）
- [x] EPLB × DeepEP 逻辑/物理分层（见 [28](./28-deepep-eplb-logical-physical-expert-map.md)）

### 下一知识点

已写 [28](./28-deepep-eplb-logical-physical-expert-map.md)；后续可接 AGRS session。
