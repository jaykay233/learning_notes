# DeepEP V1：Low-latency expert-major 打包与 `combine` 回程

> 承接 [12](./12-deepep-v1-low-latency-dispatch-kernel.md)、[13](./13-deepep-v1-low-latency-dispatch-qa.md)。本文补上讨论里尚未落盘的部分：**RECV 如何压成 expert-major**、**`row` / `src_idx` / `slot` 名词表**，以及 **`combine` 如何用 `layout_range` + `src_info` 把每一行送回原卡 token**。
>
> 代码：`internode_ll.cu` dispatch RECV（约 L352–L461）、combine（约 L715–L1138）；handle 在 `legacy.py` 的 `low_latency_combine`；`pack2`/`unpack2` 在 `utils.cuh`。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：LL 打包顺序与 combine 回程
知识点：atomicAdd 先来后到紧排；src_info[row]=源 token_idx；combine SEND 按 layout 段 + src_idx 写回；RECV 按 topk 加权求和
上次：LL dispatch 精读问答（条带 / staging / finish / count）
下次：LL 与 normal 端到端对照复习，或 internode / elastic V2
PTX / 原语：atomicAdd、unpack2、ibgda put、rdma_recv_flag、topk_weights 累加
```

## 为什么现在讲这个

dispatch 主线和 finish/count 搞清后，还缺两座桥：

1. **staging → packed**：先来后到紧排后，行号不再等于源 token，也不等于 staging slot——若不记下元数据，combine 无法回家。
2. **packed → 源卡**：combine 不能「按行号猜 src」；必须 `layout_range` 找段、`src_info` 找源 `token_idx`，写到源卡 `(global_expert, src_idx)` 槽。

混用 `row` / `slot` / `src_idx` 是读 combine 时最常见的翻车点。

## 0. 名词表（先钉死）

| 名字 | 含义 | 典型范围 |
|---|---|---|
| `local` | 本卡 local expert 下标 | `0 .. L-1` |
| `src_rank` | 数据从哪张卡来（dispatch 视角） | `0 .. P-1` |
| **`slot`** | staging `[local][src][slot]` 里的坑号（`atomicAdd` 抢） | `0 .. C-1` |
| **`row`** | expert-major `packed_recv_x[local][row]` 的行号（紧排后） | `0 .. count[local)-1` |
| **`src_idx` / 源 `token_idx`** | 源卡输入 batch 里第几个 token（消息头 → `src_info`） | `0 .. N_src-1` |

```text
staging:   rdma_recv_x[local][src_rank][slot]     ← 通信中转
packed:    packed_recv_x[local][row]              ← 给 GEMM 的 expert-major
src_info:  src_info[local][row] = 源卡 token_idx  ← 逻辑身份（≠ slot，≠ 行号本身）
```

**`src_idx` 不是 `slot`。** slot 只是中转坑；回家地址用源 token 编号。

## 1. Expert-major 怎么打包（dispatch RECV）

目标张量（host）：

```text
packed_recv_x[L, C·P, H]
packed_recv_src_info[L, C·P]
packed_recv_layout_range[L, P]   // 每格 pack(n, begin)
packed_recv_count[L]
```

对每个 `(src_rank, local_expert)`（`responsible_expert_idx` 拆出来）：

### 1.1 等 count，抢 `begin`（先来后到）

```cpp
// internode_ll.cu L407–L411
n = -rdma_recv_count[local][src] - 1;
begin = atomicAdd(packed_recv_count + local, n);
layout_range[src] = pack2(n, begin);   // (num, begin)
```

| 范围 | 顺序 |
|---|---|
| 不同 src | **先来后到**（谁先 atomicAdd 谁占更前的 `begin`） |
| 同一 src 内 | staging slot `0..n-1` → `begin+i` |

最终行序**不是**按 rank 号排序。

### 1.2 拷贝

```cpp
// L425–L436
staging[local][src][i]  →  packed_recv_x[local][begin + i]
消息头 token_idx        →  src_info[local][begin + i]
```

数值例子（local expert 1）：

```text
先到 src=2, n=2 → begin=0 → 行 0,1；src_info 例如 [5, 1]
后到 src=0, n=3 → begin=2 → 行 2,3,4；src_info 例如 [3, 7, 0]
count[1]=5
layout[2]=(2,0)  layout[0]=(3,2)
```

## 2. Combine：如何送回原来的 src rank

不靠 packed 行号猜。三件套：

```text
1. layout_range[local][dst_rank] → 这段 packed 行 [offset, offset+n) 属于该 src
2. src_info[local][row]          → 这一行对应源卡 token_idx（src_idx）
3. 写到源卡 rdma_recv_x[global_expert * C + src_idx]
```

### 2.1 SEND（expert 主机，L789–L912）

```cpp
// L790–L802, L836–L845
dst_rank = responsible_expert_idx / L;      // 要送回的卡 = 当初的 src
local    = responsible_expert_idx % L;
global_expert = rank * L + local;
unpack2(layout_range[local][dst_rank], n, offset);

for (row = offset .. offset+n) {
    src_idx = src_info[local][row];         // 源卡 token 下标
    // 读 expert 输出 x[local][row]
    // put 到 dst_rank：
    //   rdma_recv_x[(global_expert * C + src_idx) * msg]
}
```

例子：packed 行 17，`src_info[17]=5`，`layout` 说属于 rank2 → 写到 **rank2** 的槽 `(该全局 expert, token 5)`，不是槽 17。

整段发完后写门铃 `rdma_recv_flag[global_expert]=1`（L915–L928），不是再发一遍 count 人数。

### 2.2 RECV（原 token 主人卡，L942–L1135）

1. 等各相关 `rdma_recv_flag`（L950–L958）
2. `grid.sync`
3. 按**本卡原始 token 顺序**条带扫；对每个 `topk_idx[token][k]`：

```cpp
// L1046–L1047, L1074–L1114
buffer = rdma_recv_x[(global_expert * C + token_idx) * msg];
// × topk_weights[token][k] 累加进 combined_x[token]
```

与 SEND 约定同一地址：`(global_expert, 源 token_idx)`。
LL combine **在通信路径里做加权**（`topk_weights`）；normal 机内 combine 默认多是先求和。

## 3. Combine 工作流总图

```text
Expert 卡上的 x[L, C·P, H]（GEMM 输出，布局同 dispatch packed）
        │ SEND
        │ layout 取段 → src_info 取源 token → put 到各 src 卡
        │ flag[global_expert]=1
        ▼
各源卡 rdma_recv_x[expert, token]
        │ RECV
        │ 等 flag → 按本卡 token + topk 取回 → ×weight 求和
        ▼
combined_x[N, H]
```

Python 入口：`Buffer.low_latency_combine(x, topk_idx, topk_weights, handle=...)`，handle 含 `(src_info, layout_range, C, H, E)`。

## 4. 与 dispatch staging 对照

| | Dispatch SEND→RECV | Combine SEND→RECV |
|---|---|---|
| 中转 | `rdma_recv_x[local][src][slot]` | 源卡 `rdma_recv_x[global_expert][src_token]` |
| 门铃 | `rdma_recv_count = -n-1` | `rdma_recv_flag = 1` |
| 元数据 | 产出 layout / src_info | **消费** layout / src_info |
| 最终 | expert-major packed | 源卡 token-major `combined_x` |

## 5. 完整可运行核对（打包顺序 + 回程地址）

```python
#!/usr/bin/env python3
"""Simulate LL pack order and combine return addressing."""

from __future__ import annotations


def pack_expert(arrivals: list[tuple[int, list[int]]]) -> tuple[list[int], list[int], dict[int, tuple[int, int]]]:
    """arrivals: (src_rank, list of source token_idx in staging slot order)."""
    packed_tokens: list[int] = []  # values = src_idx stored in src_info
    src_info: list[int] = []
    layout: dict[int, tuple[int, int]] = {}
    for src, tokens in arrivals:
        begin = len(packed_tokens)
        packed_tokens.extend(tokens)  # stand-in for hidden rows
        src_info.extend(tokens)
        layout[src] = (len(tokens), begin)
    return packed_tokens, src_info, layout


def combine_destinations(
    local: int,
    rank: int,
    L: int,
    C: int,
    src_info: list[int],
    layout: dict[int, tuple[int, int]],
) -> list[tuple[int, int, int]]:
    """Return list of (dst_rank, global_expert, src_idx) for each packed row send."""
    global_expert = rank * L + local
    outs: list[tuple[int, int, int]] = []
    for dst_rank, (n, begin) in layout.items():
        for row in range(begin, begin + n):
            src_idx = src_info[row]
            # slot on home rank: global_expert * C + src_idx  (index in that buffer)
            outs.append((dst_rank, global_expert, src_idx))
            assert 0 <= src_idx < C
    return outs


def main() -> None:
    # FCFS: src2 first, then src0
    packed, src_info, layout = pack_expert([(2, [5, 1]), (0, [3])])
    assert packed == [5, 1, 3]
    assert src_info == [5, 1, 3]
    assert layout[2] == (2, 0) and layout[0] == (1, 2)

    # row 0 is NOT src_idx confusion with slot: staging slot was 0 for token 5, but identity is 5
    dests = combine_destinations(local=1, rank=3, L=4, C=16, src_info=src_info, layout=layout)
    # expert host rank 3, local 1 → global expert 13
    assert dests[0] == (2, 13, 5)  # home rank 2, token 5
    assert dests[1] == (2, 13, 1)
    assert dests[2] == (0, 13, 3)

    print("packed rows", packed)
    print("layout", layout)
    print("combine dests", dests)
    print("PASS")


if __name__ == "__main__":
    main()
```

期望输出含 `PASS`，且 `dests[0]==(2, 13, 5)`（行 0 → 源卡 2 的 token 5，不是「槽 0」）。

## 6. 常见错误与症状

| 误读 | 正确 | 症状 |
|---|---|---|
| packed 按 rank 排序 | 按 count 到达顺序 | 以为 rank0 总在前 |
| `src_idx == slot` | `src_idx` 是源 token；slot 是 staging 坑 | 写回错槽，reduce 错位 |
| `src_idx == row` | row 是紧排行号 | 把 17 当源 token |
| combine 按行号 put 回家 | 按 `src_info[row]` | 源卡 topk 对不上 |
| LL combine 不乘 weight | 路径内 `× topk_weights` | 和 normal 数值对不齐 |

## 7. 速记卡

```text
打包：等 count → begin=atomicAdd → staging[i]→packed[begin+i]，src_info=消息头 token
回家：layout 取段 → src_idx=src_info[row] → 源卡 (global_expert, src_idx)
RECV：本卡 token + topk_idx → 读同槽 → ×weight → combined_x
slot≠src_idx≠row
```

## 8. 自测题与答案

1. **`src_info[local][row]` 的 `row` 是什么？**
   答：该 local expert 的 expert-major 紧排行号；值才是源卡 `token_idx`。

2. **为何先来后到不影响正确性？**
   答：`layout_range` 记录每段 `(n,begin)`，`src_info` 记录每行源 token；combine 按段+src_idx 寻址，不依赖全局行序。

3. **Combine SEND 里循环变量 `token_idx` 容易和源 token 混淆，实际是什么？**
   答：packed **row**（`offset..offset+n`）；真正的源下标是 `src_idx = src_info[row]`。

4. **源卡 RECV 用什么下标读回结果？**
   答：`rdma_recv_x[topk_expert * C + 本卡 token_idx]`，与 SEND 写入的 `src_idx` 一致。

5. **LL combine 与 normal 机内 combine 在权重上的差别？**
   答：LL 在 combine kernel 内按 `topk_weights` 加权；normal 机内默认多是求和，权重常在框架侧处理。

## 9. 学习进度

- [x] LL dispatch 主流程与精读问答（12/13）
- [x] Expert-major 打包（先来后到）与 `row`/`slot`/`src_idx` 名词
- [x] LL combine：layout + src_info 回程、flag、加权 reduce

### 下一知识点

[15](./15-deepep-v1-warp-block-queue-matrix.md)：Normal×LL × dispatch/combine × intra/inter 的 warp·block·队列矩阵。
