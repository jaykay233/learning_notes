# 官方 FlashAttention 的 varlen 与 paged KV：Jagged 到底支持什么

## 本次讲解位置

```text
章节：算子开发（operators）
小节：官方 FlashAttention 的变长输入与分页 KV 接口
知识点：jagged/ragged batch 如何映射到 varlen prefill 和 paged KV decode
上次：15 · DeepSelect 的阈值筛选式精确 TopK
下次：SGLang AttentionBackend 如何在 FlashAttention、FlashInfer 与 Triton 之间 dispatch
PTX：本节不是 PTX 指令课；源码锚点是 flash_attn_interface.py 的 varlen 与 kvcache 接口
```

### 为什么现在讲这个

continuous batching 下，一个 batch 里的请求长度不同。如果误以为“官方 FlashAttention 只接受规则矩形张量”，就会在 serving 侧错误地补大量 padding；反过来，如果把 `varlen`、`paged KV` 和 FlashInfer 的 `ragged kernel` 混成同一个功能，又会误判某个 backend 或某个版本的接口边界。本节要解决的是这两个具体问题：

- prefill 如何用 `cu_seqlens` 把多条请求无 padding 地打包。
- decode 如何用 `cache_seqlens` 与 `block_table` 表示每条请求不同的 KV 长度和物理页。

## 1. 先给出结论

截至 **2026-10-20**，Dao-AILab 官方最新 release 是
[`v2.8.3.post2`](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post2)。
官方 FlashAttention 已经支持典型 serving 所需的 jagged/ragged 输入，但官方术语不是
`jagged mode`：

| 场景 | 官方接口 | 变长信息如何表达 |
|---|---|---|
| Prefill / training | `flash_attn_varlen_func` | `cu_seqlens_q`、`cu_seqlens_k` |
| Decode / incremental inference | `flash_attn_with_kvcache` | `cache_seqlens`；分页时再加 `block_table` |

稳定版源码中：

- [`flash_attn_varlen_func`](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/flash_attn/flash_attn_interface.py#L1370-L1460)
  接受 `q: [total_q, H, D]`、`k/v: [total_k, Hk, D]`，并用 `cu_seqlens_q`、`cu_seqlens_k` 切分请求。
- [`flash_attn_with_kvcache`](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/flash_attn/flash_attn_interface.py#L1464-L1560)
  接受每条请求各自的 `cache_seqlens`，并可通过 `block_table` 读取 paged KV。
- 官方稳定版测试里已经同时覆盖 `varlen` 与 paged KV：
  [`test_flash_attn_varlen_causal`](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/tests/test_flash_attn.py#L1593-L1652)。

因此，“官方有没有 jagged 开关”的答案是：没有这个开关；“官方能否处理变长请求”的答案是：能，但应按上面的官方名字理解。

## 2. 心智模型：ragged 是逻辑边界，paged 是物理地址

先把两个维度分开：

```text
ragged / varlen
  解决：一个 batch 中各请求长度不同
  数据：所有请求的 token 连续拼在一个 flat tensor 中
  边界：cu_seqlens 给出每段在 flat tensor 中的起止位置

paged KV
  解决：每条请求的 KV cache 在显存中可以不连续
  数据：固定大小的物理 page 组成 global page pool
  边界：block_table[request, logical_page] -> physical_page_id
```

`ragged` 不等于 `paged`。例如，ragged prefill 可以完全使用连续的 K/V 打包；反过来，规则长度 decode 也可以使用 paged KV。两者可以组合，但解决的是不同问题。

## 3. Varlen 的坐标计算

假设三条请求的 query 长度是：

```text
q_lens = [3, 1, 4]
```

则：

```text
cu_seqlens_q = [0, 3, 4, 8]
total_q      = 8
```

第 `r` 条请求的 Q 区间是半开区间：

```text
[cu_seqlens_q[r], cu_seqlens_q[r + 1])
```

具体为：

| request | `cu_seqlens_q` 区间 | flat Q 下标 |
|---|---|---|
| 0 | `[0, 3)` | 0, 1, 2 |
| 1 | `[3, 4)` | 3 |
| 2 | `[4, 8)` | 4, 5, 6, 7 |

Q 与 KV 的数量可以不同，因为它们由两套独立的累计边界描述：

```text
q_lens  = [3, 1, 4]  -> cu_seqlens_q = [0, 3, 4, 8]
kv_lens = [5, 2, 6]  -> cu_seqlens_k = [0, 5, 7, 13]
```

但两组边界必须表示相同数量的请求，即：

```text
len(cu_seqlens_q) == len(cu_seqlens_k) == batch_size + 1
```

否则 request 0 的 Q 无法和 request 0 的 KV 配对。

## 4. Paged KV 的地址计算

设 `page_size = 4`。某条请求的逻辑 token 位置 `p` 映射为：

```text
logical_page = p // page_size
page_offset  = p % page_size
physical_page = block_table[request, logical_page]
```

例如请求 2 有 6 个 KV token，其逻辑页是 `0, 1`。若：

```text
block_table[2] = [5, 2]
page_size      = 4
```

那么 token 位置到物理页的映射是：

| logical token | logical page | offset | physical page |
|---:|---:|---:|---:|
| 0, 1, 2, 3 | 0 | 0, 1, 2, 3 | 5 |
| 4, 5 | 1 | 0, 1 | 2 |

物理页 2 的最后两个 slot 可以属于其他请求，或暂时不用。`block_table` 只规定“这条请求的逻辑页依次指向哪些物理页”，不要求物理页连续。

## 5. 两个官方接口的 shape 与边界

### 5.1 Prefill：`flash_attn_varlen_func`

典型调用：

```python
import torch
from flash_attn import flash_attn_varlen_func

# 三条请求，H=8 个 query heads，Hk=2 个 KV heads，D=128
q = torch.randn(8, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(13, 2, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(13, 2, 128, device="cuda", dtype=torch.bfloat16)

cu_seqlens_q = torch.tensor([0, 3, 4, 8], device="cuda", dtype=torch.int32)
cu_seqlens_k = torch.tensor([0, 5, 7, 13], device="cuda", dtype=torch.int32)

out = flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=4,
    max_seqlen_k=6,
    causal=True,
)

assert out.shape == (8, 8, 128)
```

`out` 仍是 flat `[total_q, H, D]`。要回到 request 粒度，serving runtime 用 `cu_seqlens_q` 做切片：

```python
out_requests = [
    out[cu_seqlens_q[r]:cu_seqlens_q[r + 1]]
    for r in range(len(cu_seqlens_q) - 1)
]
```

稳定版 `flash_attn_varlen_func` 也接受 `block_table`，测试覆盖了 paged KV；但常见 prefill 路径仍使用连续打包的 K/V。

### 5.2 Decode：`flash_attn_with_kvcache`

`flash_attn_with_kvcache` 的外层 `q` shape 是 `[batch, q_len, H, D]`。decode 时通常 `q_len = 1`。不同请求的历史长度不由 padding 长度表示，而由 `cache_seqlens` 表示：

```python
from flash_attn import flash_attn_with_kvcache

# page_size=256，物理 KV pool 共 1024 pages
k_cache = torch.empty(1024, 256, 2, 128, device="cuda", dtype=torch.bfloat16)
v_cache = torch.empty_like(k_cache)

q = torch.randn(3, 1, 8, 128, device="cuda", dtype=torch.bfloat16)
cache_seqlens = torch.tensor([5, 2, 6], device="cuda", dtype=torch.int32)

# 每条请求只需要 ceil(cache_len / page_size) 个逻辑页
block_table = torch.tensor(
    [
        [0, 1],
        [3, 0],
        [5, 2],
    ],
    device="cuda",
    dtype=torch.int32,
)

out = flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    cache_seqlens=cache_seqlens,
    block_table=block_table,
    causal=True,
)

assert out.shape == (3, 1, 8, 128)
```

这里的 padding 逻辑页可以是任意合法页编号，只要后续有效长度的 `cache_seqlens` 不越过 `max_num_blocks_per_seq * page_size`，并且 page table 指向真实页。稳定版 FA2 的 `flash_attn_with_kvcache` 文档写明 paged KV 的 `page_block_size` 必须是 256 的倍数；使用时仍要以所装 release 的文档为准。

`flash_attn_with_kvcache` 是 inference-only 接口，不支持 backward；需要训练反向时使用 `flash_attn_varlen_func` 的 prefill/training 路径。

## 6. 当前支持不等于“所有 backend 都一样快”

官方支持 ragged 和 paged KV，只说明接口语义可用，不代表 serving 中所有工作负载都应选同一条 kernel：

| 问题 | 官方 FlashAttention | FlashInfer / 其他 backend |
|---|---|---|
| 无 padding 的变长 prefill | `flash_attn_varlen_func` | FlashInfer 有 `BatchPrefillWithRaggedKVCache` 等接口 |
| decode 的 paged KV | `flash_attn_with_kvcache(..., block_table=...)` | 专用 paged kernel 可能针对 page size、batch 形态和 GPU 代际优化 |
| 极端动态 page table | 支持接口语义 | SGLang 等 runtime 可能按 backend 选择专用实现 |

`BatchPrefillWithRaggedKVCache` 是 FlashInfer 的接口名，不是 Dao-AILab 官方 FlashAttention 的接口名。看到 SGLang 使用 ragged/paged kernel，不能据此断言它一定走的是官方 FlashAttention。

## 7. 完整可运行的 CPU trace

下面的脚本不执行 attention；它只验证 ragged packing 与 page-table 映射这两个热路径前置条件。脚本仅使用 Python 标准库，可在没有 CUDA 的机器上运行。

文件：`operators/code/flash_attention/varlen_paged_layout_demo.py`

```python
#!/usr/bin/env python3
"""CPU model of ragged token packing and paged KV address mapping."""

from __future__ import annotations


def cumulative(lengths: list[int]) -> list[int]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def unpack_ragged(values: list[int], lengths: list[int]) -> list[list[int]]:
    offsets = cumulative(lengths)
    return [
        values[offsets[request] : offsets[request + 1]]
        for request in range(len(lengths))
    ]


def paged_lookup(
    logical_values: list[int],
    page_size: int,
    block_table: list[int],
) -> list[int]:
    result: list[int] = []
    for position, expected in enumerate(logical_values):
        logical_page = position // page_size
        page_offset = position % page_size
        physical_page = block_table[logical_page]
        # page_pool maps a physical page to its slots; see main().
        value = page_pool[physical_page][page_offset]
        assert value == expected
        result.append(value)
    return result


q_lens = [3, 1, 4]
kv_lens = [5, 2, 6]
page_size = 4

q_flat = [100 + i for i in range(sum(q_lens))]
kv_flat = [200 + i for i in range(sum(kv_lens))]

cu_q = cumulative(q_lens)
cu_kv = cumulative(kv_lens)

request_q = unpack_ragged(q_flat, q_lens)
request_kv = unpack_ragged(kv_flat, kv_lens)

page_pool = {
    0: [200, 201, 202, 203],
    1: [204, -1, -1, -1],
    3: [205, 206, -1, -1],
    5: [207, 208, 209, 210],
    2: [211, 212, -1, -1],
}
request_2_pages = [5, 2]

print(f"q_lens={q_lens}, cu_seqlens_q={cu_q}")
print(f"kv_lens={kv_lens}, cu_seqlens_k={cu_kv}")
for request in range(len(q_lens)):
    print(
        f"request {request}: q={request_q[request]}, "
        f"kv={request_kv[request]}"
    )

reconstructed_kv_2 = paged_lookup(request_kv[2], page_size, request_2_pages)
print(f"request 2 logical KV={request_kv[2]}")
print(f"request 2 physical pages={request_2_pages}")
print(f"request 2 reconstructed KV={reconstructed_kv_2}")

assert cu_q == [0, 3, 4, 8]
assert cu_kv == [0, 5, 7, 13]
assert request_q == [[100, 101, 102], [103], [104, 105, 106, 107]]
assert request_kv == [[200, 201, 202, 203, 204], [205, 206], [207, 208, 209, 210, 211, 212]]
assert reconstructed_kv_2 == request_kv[2]
print("PASS: ragged boundaries and page-table mapping are consistent")
```

在 learning_notes 根目录执行：

```bash
python3 operators/code/flash_attention/varlen_paged_layout_demo.py
```

预期输出：

```text
q_lens=[3, 1, 4], cu_seqlens_q=[0, 3, 4, 8]
kv_lens=[5, 2, 6], cu_seqlens_k=[0, 5, 7, 13]
request 0: q=[100, 101, 102], kv=[200, 201, 202, 203, 204]
request 1: q=[103], kv=[205, 206]
request 2: q=[104, 105, 106, 107], kv=[207, 208, 209, 210, 211, 212]
request 2 logical KV=[207, 208, 209, 210, 211, 212]
request 2 physical pages=[5, 2]
request 2 reconstructed KV=[207, 208, 209, 210, 211, 212]
PASS: ragged boundaries and page-table mapping are consistent
```

### 逐段看 trace

| 阶段 | 输入 | 动作 | 结果 |
|---|---|---|---|
| ragged packing | `q_lens=[3,1,4]` | 累加长度 | `cu_seqlens_q=[0,3,4,8]` |
| request 切片 | flat Q 8 项 | 用相邻边界切片 | 3 / 1 / 4 个 token |
| KV packing | `kv_lens=[5,2,6]` | 使用独立边界 | `cu_seqlens_k=[0,5,7,13]` |
| paged token 0 | request 2, position 0 | `0//4=0`, physical page 5 | 取 page 5 / offset 0 |
| paged token 5 | request 2, position 5 | `5//4=1`, physical page 2 | 取 page 2 / offset 1 |

这个例子里 request 2 的逻辑页 0/1 指向物理页 5/2，物理地址不连续，但逻辑 token 顺序不变。

## 8. 本机验证边界

当前机器是 macOS。上面的 CPU layout trace 可以本地运行；官方 FlashAttention CUDA 扩展不能在 Apple GPU 上执行。CUDA 机器上可额外检查已安装版本的接口：

```bash
python3 - <<'PY'
import inspect
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

print(inspect.signature(flash_attn_varlen_func))
print(inspect.signature(flash_attn_with_kvcache))
PY
```

若安装的是 `v2.8.3.post2`，应看到：

```text
flash_attn_varlen_func(..., cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ..., block_table=None)
flash_attn_with_kvcache(..., cache_seqlens=None, ..., block_table=None, ...)
```

这只是接口级验证；真实数值仍需用 GPU 与官方测试对照。

## 9. 常见错误与症状

| 错误 | 实际含义 | 典型症状 |
|---|---|---|
| 把 `cu_seqlens` 当作每条请求长度 | 它是累计偏移，不是长度数组 | 第一条之后全部错位或越界 |
| Q/KV request 数不同 | 两套边界的第一维长度必须一致 | 配对错误、kernel 断言失败或输出语义错误 |
| `total_q` 写成最大长度 | varlen 要求 flat tensor 的总 token 数 | shape mismatch 或只处理部分请求 |
| 把 page number 当 token offset | page table 先定位物理页，再取页内 offset | 读到错误页或其他请求的数据 |
| block table 的 padding 页解引用 | 只应访问 `< cache_seqlen` 的有效 token | 可能读到未初始化或越界数据 |
| 把 varlen 当 paged | varlen 只定义逻辑边界，不定义物理页 | 误以为连续打包已经解决 cache 碎片 |
| 把 FlashInfer ragged API 当官方 FA | 两者是不同项目与 kernel 家族 | 版本、参数、性能和限制判断全部混淆 |

## 10. 自检

### Q1：`cu_seqlens_q=[0,3,4,8]` 中 request 2 的 Q 在哪一段？

**答：** 下标 `[4, 8)`，共 4 个 token。第 0、1 条分别是 `[0,3)` 与 `[3,4)`。

### Q2：Q 和 K/V 的累计长度边界可以不同吗？

**答：** 可以。每条请求的 `q_len` 与 `kv_len` 可分别在 `cu_seqlens_q`、`cu_seqlens_k` 中表示；但两者必须有相同的 `batch_size + 1` 个边界，才能逐请求配对。

### Q3：`block_table[2, 1] = 5` 表示什么？

**答：** 表示请求 2 的逻辑 page 1 指向物理 page 5。页内位置还要由 token position 对 `page_size` 取模得到。

### Q4：官方 FlashAttention 有没有一个叫 `jagged_mode` 的布尔参数？

**答：** 没有。变长 prefill 用 `flash_attn_varlen_func` 与 `cu_seqlens`；decode 的变长 KV 和分页使用 `flash_attn_with_kvcache` 的 `cache_seqlens`、`block_table`。

### Q5：官方支持 paged KV 是否意味着 SGLang 一定使用官方 FlashAttention？

**答：** 不一定。SGLang 可能按模型、GPU、dtype 和 workload 选择 FlashAttention、FlashInfer、Triton 或其他 backend；接口能力相同不代表实现路径相同。

## 进度与下一知识点

```text
[x] 官方 FlashAttention 的 varlen/ragged prefill
[x] 官方 FlashAttention 的 cache_seqlens 与 paged KV decode
[x] 区分官方 FlashAttention 与 FlashInfer ragged kernel
[x] 用 CPU trace 验证 ragged 边界和 page-table 映射
[ ] 下一知识点：SGLang AttentionBackend 的 dispatch 条件与 FA/FlashInfer/Triton 路径选择
```
