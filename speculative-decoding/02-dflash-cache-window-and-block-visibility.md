# DFlash 的 SWA cache 窗口与 draft block 可见性

> 按 DeepInfer commit [`774ce244`](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01) 解释。这里的长度和 cache 行为针对该版本 DFlash/SWA 路径；其它实现需以自己的 metadata 与 cache manager 为准。

## 本次讲解位置

```text
章节：speculative-decoding
小节：DFlash 的 SWA cache 窗口与 draft block 因果可见性
知识点：区分上下文长度、SWA 历史窗口、draft 查询数和 cache 槽位
上次：GPU 访存受限与 LPU 大 SRAM/低算力下的投机解码收益差异
下次：SWA bounded replay 与 DFlash cache 更新/回退边界
PTX：不适用
```

## 为什么现在讲这个

“上下文是 32K、SWA 窗口是 4K、draft block 是 4”描述的是三个不同维度。若把它们都叫 KV 长度，就会误以为 DFlash 每轮只输入 4K 个 token，或以为 4 个 query 都能看见未来的 draft key。本节把 cache 的驻留范围、逻辑位置和 block 内因果关系拆开。

## 1. 先建立心智模型

```text
主模型的选定层特征 ──投影──> DFlash 的历史 main key ──┐
                                                    ├─> DFlash SWA cache
本轮 draft hidden ──生成临时 draft K/V───────────────┘
                                                           │
                                      每个 query 按自己的窗口与因果索引读取
```

DFlash cache 与主模型共享的是请求的**逻辑 token 位置和寻址约定**，不是主模型原生 KV 的数值。对于 SWA，物理驻留的历史是有界窗口；绝对位置可以已经走到 32K，但不意味着 DFlash attention 必须保存或读取整段 32K 历史。

## 2. 四个容易混淆的长度

| 量 | 示例 | 含义 |
|---|---:|---|
| 主模型逻辑上下文长度 | 32K | 请求当前走到的绝对序列长度 |
| SWA `window_size` | 4K | draft attention 可查询的局部历史范围 |
| draft block 大小 `N` | 4 | 本轮 DFlash 前向里的查询位置数，由 `max_MTP_value` 决定 |
| 物理 cache 槽位数 | 不一定恰好 4K | SWA 窗口之外可能还要为回退、prefill 落盘边界和对齐留槽；由 cache manager 配置决定 |

因此，“DFlash KV 是不是 4K”更准确的回答是：**若 SWA 窗口是 4K，DFlash 的历史驻留/可见范围按这个窗口有界；实际分配不一定恰好是 4096 个槽，也不等于每轮输入 4096 个 query。**

DeepInfer 的 cache manager 将 SWA inference block size 定义为窗口加回退余量，并可再叠加 prefill 存储余量；draft metadata 的索引宽度也按 `window_size + dflash_block_size` 生成并做对齐。参见 [`cache_engine.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/cache_engine.py#L88-L110) 和 [`input_metadata/dflash.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/input_metadata/dflash.py#L36-L80)。

## 3. 主模型给 DFlash 什么

| 数据 | 来源 | 用途 |
|---|---|---|
| 选定主模型层的 attention-input 特征 | `dspark_target_layer_ids` 指定的层 | 拼成 `dflash_input`，经 DFlash 自己的 projection/norm 与 `main_kv_proj`，生成 DFlash cache 中的 main key |
| `golden_token` | 主模型验证后得到的已确定 token | 作为本轮 draft 输入的第一个 token |
| draft block 其余占位 token | DFlash 前向初始化 | 和 `golden_token` 一起形成 `N` 个查询位置；占位符不是已经接受的真实 token |
| positions、`last_valid_offset`、slot mapping、KV index | 主模型 metadata 与本轮接受结果 | 定位 request、绝对 token 位置、SWA 槽位及每个 query 的可见 key |
| 主模型 logits | 主模型验证路径 | 校验旧 draft，决定接受前缀及 correction/bonus；不是 DFlash draft attention 的直接输入 |

主模型特征先生成 DFlash 自己的 main key；DFlash draft 层也会写入本轮临时 draft K/V，源码注释说明这些 draft keys 下一轮会被覆盖。主模型自己的 KV cache 仍是另一条数据路径。

## 4. `qpos` 是绝对位置，不是 block 下标

假设本轮 block 从逻辑位置 100 开始，`N=4`：

| block 下标 `i` | 对应 `qpos` | 当前 block 中可见的 key |
|---:|---:|---|
| 0 | 100 | 100 |
| 1 | 101 | 100、101 |
| 2 | 102 | 100、101、102 |
| 3 | 103 | 100、101、102、103 |

每一行还会读取自己的 SWA 历史窗口。未来位置不可见：`qpos=100` 看不到 101；`qpos=101` 可以看到 100 和 101。窗口边界会随位置和 cache layout 变化，源码用 `dflash_kv_index` 为每个 draft query 构造索引，不是让所有 query 无差别读取同一份完整历史。

```text
query 100:  H_100 + [100]
query 101:  H_101 + [100, 101]
query 102:  H_102 + [100, 101, 102]
query 103:  H_103 + [100, 101, 102, 103]

H_i = 该 query 按 SWA 规则可见的历史 main keys；超出窗口的旧位置不读。
```

## 5. 伪代码：一轮 DFlash draft

下面是帮助理解的数据流伪代码，不是 DeepInfer 可直接运行的实现。真实代码由 fused metadata kernel 构造 positions、slot mapping 和 `dflash_kv_index`，再交给 DFlash attention kernel 消费。

```python
# 主模型处理真实 token 时，收集指定层特征并更新 DFlash 的历史 main key。
for position in main_model_token_positions:
    features = concat(
        main_attention_input[layer_id][position]
        for layer_id in dflash_target_layer_ids
    )
    dflash_hidden = projection_norm(features)
    dflash_main_keys = main_kv_proj(dflash_hidden)
    dflash_cache.write_main_keys(
        logical_positions=[position],
        keys=dflash_main_keys,
        slots=main_metadata.slot_mapping_store_swa,
    )


def make_drafts(golden_token, first_token_positions, last_valid_offset):
    N = max_MTP_value
    draft_inputs = [golden_token] + [NOISE_TOKEN] * (N - 1)

    # 由主 metadata + 接受偏移构造绝对位置和 per-query 的 SWA key 索引。
    metadata = build_dflash_input(
        first_token_positions=first_token_positions,
        last_valid_offset=last_valid_offset,
        dflash_block_size=N,
        window_size=WINDOW,
    )

    # 一个 block 内有 N 个查询行；attention 读取 DFlash cache 中被索引选中的历史
    # main keys 和本轮按因果关系可见的 draft keys。
    hidden = dflash_forward(
        input_tokens=draft_inputs,
        positions=metadata.positions,
        kv_cache=dflash_cache,
        kv_index=metadata.dflash_kv_index,
    )
    return sample_draft_heads(hidden)


# 主模型 logits 在独立的 verification 路径上校验 drafts。
# 接受结果产生下一轮 golden_token / last_valid_offset。
```

在 `N=4` 时，draft 前向处理 4 个 query 行：第一个位置输入 `golden_token`，其余位置以占位 token 初始化；它们不是 4 个已经被主模型接受的 token。真实候选由 DFlash 前向后的预测头产生。

## 6. 常见误解与症状

| 误解 | 正确理解 | 会导致的误读 |
|---|---|---|
| “主模型已经 32K，所以每个 DFlash query 读 32K KV” | SWA 路径按 `window_size` 选历史；逻辑位置走到 32K 不代表全历史都被读取 | 高估每轮带宽与 attention 计算量 |
| “DFlash cache 一定恰好等于 4K 槽” | 4K 是窗口参数，物理 cache 还受回退、prefill 余量和对齐影响 | 把逻辑窗口误当成显存张量的精确长度 |
| “N=4 就是每个 query 都能看 4 个 draft key” | N 是 block 查询行数；因果 mask 让第 i 行最多看 block 前缀，不看未来位置 | 把并行 block 误认为双向 attention |
| “DFlash 直接复用主模型 KV” | DFlash 用选定主模型层特征重新投影 main keys，并维护自己的 draft cache entries | 混淆 cache 内容与位置/slot 布局复用 |

## 自测

1. 主模型上下文为 32K、SWA 窗口为 4K，DFlash 每个 query 一定读取 32K 吗？
   答：不是；SWA query 按 4K 窗口索引历史，cache 也可用有界槽位维护历史。
2. `N=4` 表示什么？
   答：本轮 DFlash 前向有 4 个 block 查询位置，不是历史长度，也不是 4 个已经接受的 token。
3. 若 block 位置是 100、101，`qpos=100` 能看见 101 吗？
   答：不能；因果 attention 不允许读未来位置。`qpos=101` 可看 100 和 101。
4. DFlash main key 是否等于主模型 KV 中的 K？
   答：不等；它由选定主模型层特征经 DFlash 自己的投影产生，内容不同。

## 源码入口

- [DFlash 目标层特征投影与 main cache 更新](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/mtp/dflash.py#L198-L228)
- [DFlash block 大小、SWA window 和 draft 输入](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/mtp/dflash.py#L140-L145)
- [DFlash metadata 与 per-query KV index](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/input_metadata/dflash.py#L36-L80)
- [SWA cache 的历史窗口和 draft attention cache 读写](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/v4_modules/attn.py#L963-L1007)
- [SWA cache 的回退与 prefill 额外容量](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/cache_engine.py#L88-L110)
