# Attention backend 全景：KV 布局（NHD / HND / LHBNC）、架构门槛与投机解码分派

- 源码快照：vLLM `c1dc2e9cf992995953548c392f8c67db7283495b`
- 源码快照：TensorRT-LLM `0372799d1543ef5920c1318b829cf127800dbcb9`
- 示例：[attention_backend_layout_map.py](examples/attention_backend_layout_map.py)
- 相关章节：[01 vLLM token budget](01-vllm-v1-token-budget-and-spec-slots.md)（投机解码的 slot 账）、[12 KVStream](12-kvstream-multitier-kv-retrieval.md)（KV 从慢层取回）

```text
本次讲解位置
章节：inference-scheduling
小节：attention backend 的 KV 布局与 kernel 分派
知识点：NHD / HND / LHBNC 的 stride 差异、roofline 判据、spec-decode 的 reorder 分派
上次：KVStream 的多级存储检索、聚合带宽与完成驱动 I/O
下次：PD 分离两侧的布局锁与按 head 分片约束，以及跨实例 KV 传输
PTX：不适用；本节讨论显存布局、DRAM 段数与 kernel 分派条件
```

## 为什么现在讲这个

前 12 章反复处理的是「**哪些** KV 要读、**从哪**读、**什么时候**读」：token budget 决定这一步要算多少 token，KVStream 决定 storage miss 的 block 怎么从 NVMe 取回。它们都默认了一件事：KV 到了显存里之后，kernel 能以接近峰值的效率读它。

这件事并不自动成立。KV cache 在显存里怎么排（NHD 还是 HND），决定了 decode kernel 内层循环一次 `global_load` 能拿到多少**连续**字节。排错了不会算错，只会变慢；而在某些后端上根本不会让你排错——`trtllm-gen` 直接 `assert` 报错。

更有意思的是因果方向：**布局不是引擎的偏好，而是 kernel 的镜像**。同一个人写的引擎，在 H100 上跑 XQA、在 B200 上跑 trtllm-gen、在 MI300 上跑 aiter，三处的布局约束各不相同。所以「decode 该用哪个布局」这个问题，只有绑定到「这一步进哪个 kernel」之后才有答案。

这一章要建立一个判据，让你能自己回答下面三个问题，而不是背结论：

```text
1. 给定 (布局, page 形状, kernel 内层循环)，一次取子块会碎成几段？
2. 这个碎裂在这个阶段会不会被算力掩盖？（roofline）
3. 投机解码把 L 从 1 抬到 γ+1 之后，上面两个答案哪个会变、哪个不会？
```

## 1. 心智模型：四层，从逻辑形状到 DRAM

```text
逻辑形状   [L, B, H, N, C]        ← 永远是这五个维，与布局无关
    │  一个排列 permutation
物理顺序   stride                 ← 布局的名字只说这一层
    │  kernel 内层循环沿哪个维展开
一次访存   几条连续的 cache line    ← 段数与每段长度
    │  DRAM row / bank 命中
有效带宽   能不能喂满 SM          ← 与算力比值决定是否可见
```

关键点：**前两层是「表示」，后两层是「行为」**。`NHD`/`HND` 只描述第二层，本身没有优劣；优劣只在第四层才出现，而且取决于第三层是怎么写的。

## 2. 术语：五个字母分别是什么

逻辑形状恒为 `[L, B, H, N, C]`，与 `vllm/v1/kv_cache_layout.py:18-28` 一致：

| 字母 | 含义 | 典型值（Llama-3-8B） |
|---|---|---|
| `L` | layer 数（每层一份 KV） | 32 |
| `B` | block / page 数（PagedAttention 的分配单位） | 数万，按需分配 |
| `H` | **num_kv_heads**（GQA 下远小于 query head 数） | 8 |
| `N` | **每个 page 里的 token 数** = `block_size` | 16 |
| `C` | head_dim（= `hidden_size / num_heads`） | 128 |

`N` 这个名字来自 attention 两个 GEMM 的矩阵维度命名：`S = Q·Kᵀ` 里 `M` 是 query 行数、`N` 是 key 列数。KV 在 K 侧是「被乘的矩阵」，所以它的 token 维就叫 `N`。vLLM 把它推广成 `num_states` 以兼容 Mamba/SSM（`vllm/envs.py:1856-1859`）。

> **陷阱**：不少博客把 `N` 解释成 num_heads。判据只有一条——看 stride 第二项是不是 `block_size * head_dim`。别从字母猜。

`N` 还有一个容易忽略的身份：它同时是**分页粒度的旋钮**（`--block-size`）。后面的分段数直接正比于它。

## 3. 布局就是 stride 排列：六种，两个别名

`KVCacheLayout` 的枚举值就是一个排列，把逻辑轴映射到物理顺序（`vllm/v1/kv_cache_layout.py:23-28`）：

| 枚举 | 排列 | 物理顺序 | FlashInfer 别名 | 页内 `H` 的位置 |
|---|---|---|---|---|
| `LBHNC` | (0,1,2,3,4) | L, B, H, N, C | **HND** | 在 `N` 外 |
| `LBNHC` | (0,1,3,2,4) | L, B, N, H, C | **NHD** | 在 `N` 内 |
| `LHBNC` | (0,2,1,3,4) | L, H, B, N, C | 无 | 在 `B` 外（跨 page） |
| `BLHNC` | (1,0,2,3,4) | B, L, H, N, C | HND | 页外再按 B 排 |
| `BLNHC` | (1,0,3,2,4) | B, L, N, H, C | NHD | 同上，页内 NHD |
| `BHLNC` | (1,2,0,3,4) | B, H, L, N, C | HND | 又是另一种外层嵌套 |

所以 `NHD`/`HND` 只是**页内**二维的简称。`LHBNC` 是第三种形态：`H` 跑到 `B` 外面去了，意味着**一个 head 的所有 page 在显存里连成一片**。

这不是理论构造。`vllm/v1/attention/backends/rocm_attn.py:232-237` 的注释直接点明它的用途：

```text
# The native HIP kernels hardcode group-contiguous block addressing, so
# they need the K and V groups to span all blocks (H outermost within the layer)
```

## 4. 量化：一页 32 KiB，碎成几段

取典型配置 `H=8, N=16, C=128, fp16`，一页 `8*16*128*2 = 32768 B = 32 KiB`。三种布局都是**一整块连续内存**，差别只在取子块时：

```text
$ cd inference-scheduling/examples && python3 attention_backend_layout_map.py
==============================================================================
[1] KV 布局与子块分段   page=[H=8, N=16, C=128] 2B/elem, 历史 4 页
==============================================================================
一页总字节 = H*N*C*elem = 8*16*128*2 = 32768 B = 32 KiB（三种布局都是一整块连续内存）

NHD   = LBNHC
  stride(byte)  H=256  B=32768  N=2048  C=2
  decode 单页: 1 head × N × C        -> 16 段 × 每段   256 B
  decode 扫 4 页: 1 head × B × N × C -> 64 段 × 每段   256 B
  prefill 单页: 1 token × H × C      ->  1 段 × 每段  2048 B
  搬单页: H × N × C                   ->  1 段 × 每段 32768 B

HND   = LBHNC
  stride(byte)  H=4096  B=32768  N=256  C=2
  decode 单页: 1 head × N × C        ->  1 段 × 每段  4096 B
  decode 扫 4 页: 1 head × B × N × C ->  4 段 × 每段  4096 B
  prefill 单页: 1 token × H × C      ->  8 段 × 每段   256 B
  搬单页: H × N × C                   ->  1 段 × 每段 32768 B

LHBNC
  stride(byte)  H=16384  B=4096  N=256  C=2
  decode 单页: 1 head × N × C        ->  1 段 × 每段  4096 B
  decode 扫 4 页: 1 head × B × N × C ->  1 段 × 每段 16384 B
  prefill 单页: 1 token × H × C      ->  8 段 × 每段   256 B
  搬单页: H × N × C                   ->  8 段 × 每段  4096 B
```

四行读法：

1. **搬整页三者相同**（32768 B 一段）。这是为什么 PD 分离搬运、KV offload 写回都不受布局影响——它们按页搬。
2. **decode 单页**：NHD 碎成 16 段 × 256 B，HND 是 1 段 × 4096 B。256 B 恰好是两个完整 L2 line（128 B），所以**没有一个字节被浪费**，损失纯粹来自「16 个请求 vs 1 个请求」和 DRAM row 局部性。
3. **decode 跨页**：NHD 64 段，HND 4 段（每页一段），LHBNC 1 段 16384 B。这一段连续长度是 `B*N*C*2`，即「一个 head 的整段历史」在显存里是连续的——aiter 的 asm kernel 要的就是这个。
4. **prefill 单页**：反过来，NHD 1 段 2048 B，HND 8 段。因为 prefill 的 tile 是「一段 token × 若干 head」，页内本来就是 token 在外。

顺带给出一个常常被忽略的推论：`N` 越大，NHD 的段越长、惩罚越小。`N` 从 16 拉到 128 时每段变成 2048 B，NHD 与 HND 的 DRAM 效率差距基本消失，HND 剩下的好处只有地址算术更简单。反之 `N=4` 时每段只有 64 B，NHD 会退化成半个 cache line，浪费一半带宽。

## 5. 为什么 decode 在乎、prefill 不在乎：把 roofline 算出来

不要停在「decode 是带宽 bound」这句话上，把系数推出来。

设：`L` 个 query token、`S` 个 KV token、`H_q = G·H_kv` 个 query head、head_dim `D`、元素 `e` 字节。

```text
FLOPs   = 2·L·S·D·H_q   (QKᵀ)
        + 2·L·S·D·H_q   (P·V)
        = 4·L·S·D·G·H_kv

KV 字节 = 2(K 和 V 各一份)·S·D·H_kv·e
        = 2e·S·D·H_kv

强度    = 4·L·S·D·G·H_kv / (2e·S·D·H_kv) = 2LG/e  FLOP/B
```

取 `e = 2`（fp16/bf16）→ **强度 = `L·G` FLOP/B**。H100 的拐点：

```text
ridge = 989 TFLOPS / 3.35 TB/s = 295 FLOP/B
```

| 模型 | `G` | L=1 | L=2 | L=4 | L=8 |
|---|---|---|---|---|---|
| Llama-3-8B 32/8 | 4 | 4 | 8 | 16 | 32 |
| Llama-3-70B 64/8 | 8 | 8 | 16 | 32 | 64 |
| Mistral-7B 32/8 | 4 | 4 | 8 | 16 | 32 |
| DeepSeek MLA | 1 | 1 | 2 | 4 | 8 |

跨过 `ridge` 需要的 `L`：`G=4` 要 74，`G=8` 要 37，`G=1`（MLA）要 295。

> **注意系数**：因为 K 和 V 各占一份字节，强度是 `L·G` 而不是 `2LG`。很多口算稿漏掉这个 1/2，把「需要多少个 draft token」低估一半。

物理含义：一个 KV 元素从 HBM 读进来只做 `G·L` 次乘加，而 H100 每读 1 字节可以做 295 次乘加。**中间差了两个数量级**，意味着 attention 在 decode 阶段有大量时间在等 HBM，任何能减少 DRAM 请求的手段都是净收益；反过来 prefill 有 `L=512..4096`，强度 1024~8192 FLOP/B，高出拐点 3.5×~28×，访存碎裂被矩阵乘完全掩盖，所以**布局对 prefill 几乎不重要**。

这就是那句流行的「decode 适合 HND，prefill 适合 NHD」的完整依据。但它的准确表述是：

```text
prefill 不在乎布局 → 留在默认的 NHD（也是 K/V projection 输出 [M,H,D] 的自然形状）
decode 在乎      → 被专门优化成 HND
```

## 6. 投机解码改变了什么：只改 `L`

朴素 decode 每步 `L=1`，投机解码的 verify 步是 `L = γ+1`（γ 个 draft token + 1）。

| | 朴素 decode | verify 步 |
|---|---|---|
| 每步读的 KV | 整段历史 | **整段历史（不变）** |
| 每个 KV 字节被读次数 | 1 / layer / request | **1 / layer / request（不变）** |
| 每个 KV 字节被复用次数 | `G` | `G·(γ+1)` |
| 强度（fp16） | `G` | `G·(γ+1)` |

只有最后一行变了。所以：

- **「N 必须连续」这个要求在 NHD 下一点没被放宽**：仍然是 16 段 × 256 B 跨 2048 B 跳。
- HND 省下的 DRAM 事务数完全一样，收益不会消失。
- 被削弱的只是**相对权重**：`L` 变大后每个 tile 上有更多 MMA 可以掩盖访存延迟，同一次 strided load 占的时间比例变小。

### 6.1 那为什么实践中真的会变？因为 kernel 分派

vLLM 有一个显式开关，恰好就是答案（`vllm/v1/attention/backend.py:655-674`）：

```python
self.reorder_batch_threshold = reorder_batch_threshold
if self.reorder_batch_threshold is not None and supports_spec_as_decode:
    # If the backend supports spec-as-decode kernels, then we can set
    # the reorder_batch_threshold based on the number of speculative
    # tokens from the config.
    speculative_config = self.vllm_config.speculative_config
    if (speculative_config is not None
            and speculative_config.num_speculative_tokens is not None):
        self.reorder_batch_threshold = max(
            self.reorder_batch_threshold,
            max_decode_query_len(self.vllm_config),
        )
```

`max_decode_query_len` 就是 `1 + num_speculative_tokens`（`vllm/v1/attention/backend.py:569-576`）。于是分出两条路：

| 后端有 spec-as-decode kernel？ | `reorder_batch_threshold` | verify 步去哪 | 布局 |
|---|---|---|---|
| 有（XQA / trtllm-gen / aiter / hpc） | `1+γ` | 留在 decode 路径，`q_len_per_req=γ+1` | 沿用该 decode kernel 的布局 |
| 没有（原生 FA decode / Triton / TurboQuant） | `1` | 被划进 **prefill 组** | 走 prefill kernel 的布局 |

最直白的证据在 FlashInfer 的 DCP 分支——刻意把阈值压回 1，让 spec query 走 prefill 路径（`vllm/v1/attention/backends/flashinfer.py:928-940`）：

```text
# trtllm-gen decode receives no cp_rank/global-seq-len information,
# so its end-aligned causal mask is wrong for q_len > 1 over the
# DCP-interleaved local KV shard ... Keep the threshold at 1 under DCP
# so spec queries take the DCP-aware prefill path
```

反过来，XQA 明确支持多 token query（带 draft mask，`flashinfer.py:1208-1217`）。所以同一个引擎，装了 `FLASHINFER + XQA` 的 H100 在投机解码下**仍然是 HND**；而 SM100 上 trtllm-gen 连 prefill 都要 HND。

### 6.2 反向效应：写路径随 `γ` 变好

verify 步要一次写 `γ+1` 个 token 的 KV。HND 下这是 `H` 段 × `(γ+1)·C·2` B：

| γ | HND 写入分段 | NHD 写入分段 |
|---|---|---|
| 0 | 8 × 256 B | 1 × 2048 B |
| 3 | 8 × 1024 B | 1 × 8192 B |

所以「投机解码 ⇒ 更该 NHD」这个推论在写路径上不成立，`γ` 越大 HND 的写入反而越不碎。

## 7. 四家 kernel：XQA / trtllm-gen / aiter / hpc

这四个名字常被并列当成「一类东西」，其实跨了硬件厂商和抽象层次。

| 名字 | 归属 | 架构门槛 | 负责阶段 | 布局 | spec-as-decode |
|---|---|---|---|---|---|
| **trtllm-gen** | NVIDIA TRT-LLM 代码生成 kernel 家族 | SM100/103/107 | decode + prefill | **强制 HND** | 是（device 端裁 draft） |
| **XQA** | NVIDIA TRT-LLM `cpp/kernels/xqa` | SM90 / SM120 家族 | decode（MQA/GQA、beam） | HND 原生；TMA 支持 NHD | 是（`SPEC_DEC` 分支） |
| **aiter** | AMD AITER 算子库 | gfx9 CDNA3+ | decode（asm）+ prefill | `LHBNC` / `SHUFFLE` | 是（MTP kernel） |
| **hpc** | 腾讯 `hpc-ops` | SM90 only | decode(MTP) + prefill | **`LBNHC` = NHD** | 是（mtp ≤ 3） |
| FlashInfer native | vLLM | SM80+ | decode + prefill | 任意（stride-aware） | 否 → 退回 prefill |
| FA3 / FA4 | vLLM | SM90 / SM100 | prefill 为主 | 任意（stride-aware） | 不切分，varlen 一把跑 |

### 7.1 XQA

官方定位一句话：`# XQA - A set of optimized kernels for generation-phase MQA/GQA`。它就是为「已有长 KV、只算少量 query token」写的，也是 `L` 这个维度的来源。

- SM90 默认走 `mha_sm90.cu`（TMA + QGMMA），可强制退回标量版 `mha.cu`（`XQA_USE_QGMMA=0`）。
- `XQA_NB_SUB_SEQ` 控制「一个 KV head 用几个 thread block」——就是 split-KV，用来在 batch 小、head 少时填满 SM。
- 原生 page 排法是 HND：`cpp/kernels/xqa/test/refAttention.h:52-63` 的 `else` 分支是 `pageIdx, idxHead, pos` 顺序。
- 但它**后来专门加了 NHD 路径**，靠 TMA 描述符：`cpp/kernels/xqa/tensorMap.cpp:56-68` 里 `globalStrides[token] = headElems * nbKHeads * elemBytes`，一个 box 就把「跨 head 跳」的 NHD 读成连续 tile。

**所以「XQA 必须 HND」这个说法不成立**，它两种都能跑。vLLM 里 XQA 路径也确实没有布局断言（只有 trtllm-gen 有）。这也回过头证明了「TMA 让 NHD 的惩罚基本消失」——不是理论，是 XQA 真的为此写了一条路径。

在 vLLM 里 XQA 的准入门槛写得很具体（`flashinfer.py:636-637`）：`16 <= head_dim <= 256 and head_dim % 16 == 0`。另外三个条件会把你踢回原生 FlashInfer decode：head_dim 不支持、需要返回 LSE（DCP）、非 causal。

### 7.2 trtllm-gen

最容易被误会的名字：它不是 TensorRT-LLM 这个引擎，也不是某个手写 kernel，而是**一套内部 DSL 生成的 cubin 家族**。TensorRT-LLM 文档里两处说得很清楚：

```text
On Blackwell we use the kernel generated from an internal code-gen based solution called `trtllm-gen`.
The trtllm-gen path (with pre-built CUBIN integration) is not covered here.
```

Python DSL 按 `(arch, head_dim, page_size, mask, dtype, sparse)` 枚举分支 → 生成 CUDA → 编成一堆 cubin → 运行时按 shape **查表**选一个。两个工程后果：

1. **编译期特化**：`--cuda_architectures` 没包含当前 SM 时，整个 cubin 表被 `-DEXCLUDE_SM_<arch>` 编空，运行时直接报「this build has none」。
2. **形状约束变成硬门槛**：`page_size >= 128` 必须用它（`flashinfer.py:1430`），反过来它要求 HND（`flashinfer.py:2290` 的 `assert`）。

它也是唯一支持**device 端裁 draft** 的内核：adaptive verification 会在 GPU 上改 `qo_indptr`，所以要 varlen（`flashinfer.py:1136-1150` 的注释：*"only trtllm-gen supports that"*）。

### 7.3 aiter

**先把名字拆开**：AITER 不是某个 attention kernel，而是 AMD 的算子库整体（形态类似 cuBLAS + CUTLASS + FlashInfer 的合集），CK 模板、手写 assembly、Triton 三套并存。vLLM 里叫 `rocm_aiter_ops`，attention 只是它的一部分。

它的 attention decode 走手写 asm，配置表按 slot 维护（`rocm_aiter_fa.py:413-418`）：

```text
"""Dense per-slot scale tables the asm decode kernel reads for an fp8 cache.
One table per distinct (device, shape, scale), shared by every layer and
every metadata builder in the process, so the target and draft models with
```

「target and draft models」——这句说明它就是为**投机解码（draft model 同进程）**准备的。

布局上它有两个特殊值：`LHBNC` 和 **`SHUFFLE`**。后者是第三种布局，超出了 NHD/HND 的二维框架：把 K/V 同一 `(block, head)` 的数据物理交错排，让 asm kernel 一次 load 同时拿到 K 和 V（`rocm_aiter_fa.py:892-905` 的注释解释了为什么必须是 `H` 最外：*"pa_fwd_asm strides between pages by a whole dense page, so the two head slots ... have to be separate planes spanning every block"*）。

回到第 4 节那张表：`LHBNC` 下「一个 head 扫 4 页历史」是 **1 段 16384 B**，比 HND 的 4 段更好。代价在最后一行——「搬单页」变成 8 段 × 4096 B，所以这种布局不利于按页整体搬运的 PD 传输。

### 7.4 hpc

腾讯的 `hpc-ops`，也是最该记住的**反例**。约束很硬：只认 SM90、`head_dim` 固定 128、`block_size` 固定 64、纯 attention（不含 RoPE/RMSNorm，把 RoPE+Norm 拆成独立 pass 并把 FP8 scale 透传下来，`hpc_attn.py:270-322`）。

但它有 MTP decode kernel，而且布局是 NHD：

```text
class HpcAttentionBackend(AttentionBackend):
    """HPC attention backend (pure attention, no RoPE/Norm).

    KV cache layout: LBNHC (num_blocks, num_kv_heads, block_size, 2 * head_size).
    """
```

对投机的支持是显式的（`hpc_attn.py:125-138`）：

```text
# MTP constraint: HPC decode kernel only supports mtp in {0, 1, 2, 3}
raise ValueError(
    f"HPC attention only supports up to 3 speculative tokens "
    f"(mtp ∈ {{0, 1, 2, 3}}), got "
    f"num_speculative_tokens={spec_config.num_speculative_tokens}. ")
```

它还做了 dynamic schedule：先按 seq_len 把 decode 请求切成任务图（`task_map`，`min_split_len=1024`），kernel 按图吃掉，避免 batch 内长短请求严重不均导致尾部空转。

**hpc 证明了「spec decode ⇒ HND」是伪相关。** 正确的因果链只有三环：

```text
decode → 一请求一 KV head 顺序流式扫描的专用 kernel → 需要 N 连续
                  ↑
        真正决定布局的是这一环的内层循环，以及它靠什么掩盖 stride
```

## 8. 布局是全局唯一的，这带来三个工程约束

引擎启动时只为整个模型解析出**一个**布局（`vllm/v1/attention/backends/utils.py:265-277`）：

```text
Runs once in the engine core. Every worker reports the layouts its backends
support, most preferred first; all ranks run the same backends, so their lists
must agree. ... An explicit VLLM_KV_CACHE_LAYOUT must be one of the candidates
or resolution fails, with the legacy NHD/HND names as aliases for LBNHC/LBHNC
```

所以不能「prefill 用 NHD、decode 用 HND」同时存在。三个直接后果：

1. **PD 分离必须两侧一致**。布局不同就没法整行搬运，NIXL 会拒绝而不是帮你转置（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:2419-2423`）：

   ```text
   # Whole rows transfer as one unit, so both sides must pack them alike.
   raise RuntimeError(
       "Packed KV block rows differ between P and D. Use the same "
       "VLLM_KV_CACHE_LAYOUT and block size on both sides.")
   ```

2. **head 分片依赖布局**。SGLang 里 MiniMax 的报错说明 `head-major page rows ... cannot be resliced by attention head`，于是 PD 两侧 attention TP 必须相等。

3. **block 切分依赖布局**。`--block-size` 与 kernel block 不整除时，报错会提示换 layer-compact 布局（`vllm/v1/worker/utils.py:410-418`）：

   ```text
   f"... which packed {spec.block_size}-token KV cache blocks cannot be split into. "
   f"Reduce --block-size to {self.kernel_block_size} or set VLLM_KV_CACHE_LAYOUT "
   f"to a layer-compact layout (e.g. LBNHC)."
   ```

## 9. 怎么查、怎么验

### 9.1 现场扫描：换版本后事实会变

脚本的 `--scan` 模式直接读源码，不依赖文档记忆：

```bash
$ python3 attention_backend_layout_map.py --scan /path/to/vllm
==============================================================================
[4] 现场扫描 .../vllm/v1/attention/backends
==============================================================================
b12x.py                                              LBHNC, BLHNC
composite.py                                         (未声明 = 任意布局)
cpu_attn.py                                          LBHNC
flashinfer.py                                        LBHNC, BLHNC
flex_attention.py                                    LBNHC
hpc_attn.py                                          LBNHC
rocm_aiter_fa.py                                     LHBNC, LBHNC
rocm_aiter_unified_attn.py                           LBHNC, LHBNC
rocm_attn.py                                         LHBNC, LBHNC
turboquant_attn.py                                   LBNHC
mla/indexer.py                                       BLHNC
mla/rocm_aiter_mla_sparse.py                         LBNHC, LBHNC, BLHNC

共 12 个 backend 声明了布局偏好；未声明者读 stride，不读布局名。
```

注意 `flash_attn.py` 不在列表里——它不声明，也就是**任意布局都能跑**。这正是「FA 系根本不区分 decode/prefill」的机制来源：它读 stride，不读布局名，整个 batch 一个 varlen 调用，只传 `cu_seqlens_q`。

### 9.2 运行时确认选的是哪个 kernel

分派逻辑本身很短（`flashinfer.py:1152-1160`）：

```python
@staticmethod
def _get_flashinfer_trtllm_api_decode_kernel() -> FlashInferDecodeKernel:
    if current_platform.is_device_capability(90) or \
            current_platform.is_device_capability_family(120):
        return FlashInferDecodeKernel.XQA
    assert current_platform.is_device_capability_family(100)
    return FlashInferDecodeKernel.TRTLLM_GEN
```

一句话记：**H100/H200 → XQA，B200/GB200 → trtllm-gen，MI300 → aiter，H100 且装了腾讯优化 → hpc。**

代码里永远验 stride，不验名字：

```python
k = kv_cache[0]
print(k.shape, k.stride())
# HND: 倒数第二维 stride == head_dim*2（页内 token 连续）
# NHD: 倒数第二维 stride == num_kv_heads*head_dim*2
```

## 10. 常见错误与症状

| 错误 | 症状 | 原因 |
|---|---|---|
| 把 `N` 当 num_heads | 手算的分段数与实测差一个 `H` 倍 | `N` 是 page 内 token 数；判据看 stride |
| 认为「spec decode ⇒ 必须 HND」 | 在 hpc / FA 系上白改配置 | hpc 有 MTP kernel 却用 NHD；决定因素是 kernel 内层 |
| 以为 `NHD` 会丢数据或算错 | 没有任何精度异常，只是慢 | 布局只影响访存段数，不影响数值 |
| 用 `2LG` 估算术强度 | 「需要多少 draft token」低估一半 | K 和 V 各占一份字节，强度是 `2LG/e` = `LG`（fp16） |
| 在 PD 两侧用不同 `VLLM_KV_CACHE_LAYOUT` | 启动或首包传输就报错，不是静默错 | 整行搬运要求两侧打包一致 |
| 为 decode 优化把布局改成 HND，但 prefill 占 90% 时间 | 端到端没有变化甚至更慢 | prefill 强度高出拐点 3.5~28×，布局收益被算力掩盖 |
| 在 `N=4` 的 page 上期待 NHD 够用 | 有效带宽接近腰斩 | 每段只有 64 B，不足半个 cache line |
| 在 SM100 上关掉 trtllm-gen 想用 NHD | `assert ... == "HND"` 直接失败 | trtllm-gen 的 prefill 也要求 HND |
| 假设自建 build 一定有 kernel | 报「requires a fused context FMHA kernel, and this build has none」 | `--cuda_architectures` 漏了当前 SM，cubin 表被编空 |

## 11. 自检

**Q1：`N` 是什么？为什么不是 num_heads？**

`N` 是每个 page 里的 token 数，等于 `--block-size`。名字来自 `S = Q·Kᵀ` 的列维（key 侧 token 数）。判据是 stride：页内 `N` 的 stride 等于 `H*C`（NHD）或 `C`（HND）；`H` 的 stride 才是 `N*C` 或 `C`。

**Q2：一页 32 KiB 在 NHD 和 HND 下哪个占的内存少？**

一样。两者都是 32768 B 的连续块，只是内部排列不同。布局不改变容量，只改变子块的连续性——`kv_cache_layout.py` 里的六个枚举物理字节数完全相同。

**Q3：投机解码 `γ=3` 时 decode 一定不用 HND 了吗？**

不一定。形状上仍是「1 个 head 扫整段历史」，HND 的收益完全保留。变的只是 `L`：强度从 `G` 升到 `G(γ+1)`，`G=8` 时 8 → 32，仍比 295 的拐点低 9 倍。只有当后端**没有** spec-as-decode kernel（`reorder_batch_threshold` 保持 1）时，verify 步才会被划进 prefill 组、改用 prefill kernel 的布局。

**Q4：为什么 trtllm-gen 在 SM100 上连 prefill 都要 HND，而 FA3 两种都行？**

trtllm-gen 是编译期特化的 cubin 表，地址计算和内层循环在生成时写死，只认一种；FA 系读 stride，靠 TMA/描述符在运行时处理步长，所以布局无关。这也是为什么 trtllm-gen 换布局只能重新编译，而 FA 换布局只是换个 stride。

**Q5：`LHBNC` 比 `HND` 更适合 decode，为什么不是默认？**

因为它把 `H` 放到了 `B` 外面：一个 head 的整段历史连续（1 段 16384 B），但「搬一整页」变成 8 段 4096 B。PD 分离、KV offload、prefix cache 拷贝都按页搬，代价太高。`LHBNC` 只在 kernel 真的按 head 连续寻址时（aiter 的 asm decode）才划得来。

## 12. 一页速查

```text
逻辑形状恒为 [L, B, H, N, C]
N = 每页 token 数 = block_size；H = num_kv_heads；C = head_dim

NHD  = LBNHC = L,B,N,H,C  → 「一个 token 的全部 head」连续  → prefill 默认
HND  = LBHNC = L,B,H,N,C  → 「一个 head 的整段历史」连续    → decode 常用
LHBNC        = L,H,B,N,C  → head 跨 page 最外              → aiter asm

一页 32 KiB（H=8,N=16,C=128,fp16）三种布局字节数相同
  decode 单页 1 head: NHD 16段×256B | HND 1段×4096B | LHBNC 1段×4096B
  decode 跨 4 页 1 head: NHD 64段 | HND 4段 | LHBNC 1段×16384B
  搬单页: NHD 1段 | HND 1段 | LHBNC 8段  ← LHBNC 的代价

强度 = 2LG/e FLOP/B（fp16 → LG）；H100 ridge = 295 FLOP/B
  decode L=1,G=8 → 8（低 37×，纯带宽 bound）
  prefill L=2048,G=8 → 16384（高 55×，算力 bound）
  注意：不是 2LG，K 和 V 各占一份字节

投机解码只改 L：1 → γ+1；每 KV 字节的 DRAM 访问次数不变
  有 spec-as-decode kernel → reorder_batch_threshold=1+γ，留在 decode 路径
  没有                     → threshold=1，verify 步划进 prefill 组

H100/H200 → XQA        B200/GB200 → trtllm-gen
MI300     → aiter      SM90+腾讯  → hpc（NHD，mtp<=3）
trtllm-gen 强制 HND（prefill 也是）；XQA/FA 两种都行（TMA/stride-aware）

布局全局唯一；PD 两侧必须一致；改 block_size 可能触发 kernel block 切分报错
```

## 附录：完整脚本

```python
#!/usr/bin/env python3
"""Attention backend 的 KV 布局 / 架构门槛 / spec-decode 分派对照表。

只用标准库，不需要 GPU。默认运行三张表：

    1. NHD / HND / LHBNC 三种布局的 stride，以及「取一个子块要分成几段」
    2. decode 与 prefill 的算术强度，以及相对 roofline 拐点（ridge）的倍率
    3. vLLM attention backend 的布局声明与架构门槛（源码快照表）

带 --scan 时第 3 张表改为从 vLLM 源码现场抽取，用来核对换版本后的事实。

    python3 attention_backend_layout_map.py
    python3 attention_backend_layout_map.py --scan /path/to/vllm
"""

import argparse
import itertools
import os
import re
import sys

# ---------------------------------------------------------------------------
# 1. 布局：逻辑形状恒为 [L, B, H, N, C]，值是该顺序映射到物理（内存）顺序的排列
#    与 vllm/v1/kv_cache_layout.py 的 KVCacheLayout 一致。
# ---------------------------------------------------------------------------
LOGICAL = ("L", "B", "H", "N", "C")

LAYOUTS = {
    "NHD   = LBNHC": (0, 1, 3, 2, 4),  # L, B, N, H, C
    "HND   = LBHNC": (0, 1, 2, 3, 4),  # L, B, H, N, C
    "LHBNC": (0, 2, 1, 3, 4),  # L, H, B, N, C（head 跨 block 最外）
}

# 典型形状：kv_head 数、每 page token 数、head_dim，以及要扫的历史 page 数
DEFAULT_PAGE = {"H": 8, "N": 16, "C": 128}
DEFAULT_BLOCKS = 4
ELEM_BYTES = 2  # fp16 / bf16


def strides(layout_perm, sizes):
    """返回 {dim: stride_in_elements}，按「最内层 stride=1」反向累乘。"""
    stride = {}
    acc = 1
    for dim_index in reversed(layout_perm):
        dim = LOGICAL[dim_index]
        stride[dim] = acc
        acc *= sizes[dim]
    return stride


def segments(layout_perm, sizes, vary, fixed):
    """把子块展开成字节偏移集合，返回 (段数, 每段字节数)。

    vary:  参与变化的逻辑维，例如 ("N", "C") 表示「一个 head 的整页 token」
    fixed: 固定的逻辑维取值，例如 {"B": 0, "L": 0, "H": 3}
    """
    stride = strides(layout_perm, sizes)
    offsets = []
    ranges = [range(sizes[dim]) for dim in vary]
    base = sum(fixed.get(dim, 0) * stride[dim] for dim in LOGICAL if dim not in vary)
    for coords in itertools.product(*ranges):
        offsets.append(base + sum(c * stride[dim] for c, dim in zip(coords, vary)))
    offsets.sort()
    runs = 1
    for prev, cur in zip(offsets, offsets[1:]):
        if cur != prev + 1:
            runs += 1
    return runs, (len(offsets) // runs) * ELEM_BYTES


def report_layouts(page, elem_bytes, num_blocks):
    sizes = dict(page)
    sizes.update({"L": 1, "B": num_blocks})
    page_bytes = page["H"] * page["N"] * page["C"] * elem_bytes
    print("=" * 78)
    print(f"[1] KV 布局与子块分段   page=[H={page['H']}, N={page['N']}, C={page['C']}] "
          f"{elem_bytes}B/elem, 历史 {num_blocks} 页")
    print("=" * 78)
    print(f"一页总字节 = H*N*C*elem = {page['H']}*{page['N']}*{page['C']}*{elem_bytes}"
          f" = {page_bytes} B = {page_bytes / 1024:.0f} KiB（三种布局都是一整块连续内存）")
    print()

    for name, perm in LAYOUTS.items():
        st = strides(perm, sizes)
        print(f"{name}")
        print("  stride(byte)  " + "  ".join(
            f"{d}={st[d] * elem_bytes}" for d in ("H", "B", "N", "C")))

        patterns = [
            ("decode 单页: 1 head × N × C", ("N", "C"), {"B": 0, "L": 0, "H": 3}),
            (f"decode 扫 {num_blocks} 页: 1 head × B × N × C", ("B", "N", "C"),
             {"L": 0, "H": 3}),
            ("prefill 单页: 1 token × H × C", ("H", "C"), {"B": 0, "L": 0, "N": 5}),
            ("搬单页: H × N × C", ("H", "N", "C"), {"B": 0, "L": 0}),
        ]
        for label, vary, fixed in patterns:
            runs, seg_bytes = segments(perm, sizes, vary, fixed)
            print(f"  {label:<32} -> {runs:>2} 段 × 每段 {seg_bytes:>5} B")
        print()

    print("判据：段数越少、每段越长，DRAM 请求数越少。整页搬运时三者完全相同，")
    print("      差别只出现在 kernel 内层循环沿哪个维展开、以及是否跨 page。")
    print()


# ---------------------------------------------------------------------------
# 2. 算术强度：一个 decode step 里每个 KV 字节被乘加几次
# ---------------------------------------------------------------------------
def report_roofline(tflops, bw_tbs, gqa, elem_bytes):
    ridge = tflops * 1e12 / (bw_tbs * 1e12)
    print("=" * 78)
    print(f"[2] 算术强度与 roofline   {tflops:.0f} TFLOPS / {bw_tbs:.2f} TB/s")
    print("=" * 78)
    print("推导：L 个 query token、S 个 KV token、H_q=G*H_kv 个 query head、head_dim D")
    print("  FLOPs = 2*L*S*D*H_q (QK^T) + 2*L*S*D*H_q (P·V) = 4*L*S*D*G*H_kv")
    print(f"  KV 字节 = 2(K,V)*S*D*H_kv*{elem_bytes}B = {2 * elem_bytes}*S*D*H_kv")
    print(f"  => 强度 = 4*L*S*D*G*H_kv / ({2 * elem_bytes}*S*D*H_kv)"
          f" = {4 / (2 * elem_bytes):.1f}*L*G  FLOP/B")
    print(f"  ridge = {tflops:.0f}e12 / {bw_tbs}e12 = {ridge:.0f} FLOP/B "
          f"(超过即算力 bound)")
    print()
    header = "  ".join(f"L={L:<5}" for L in (1, 2, 4, 8))
    print(f"{'模型 (H_q/H_kv)':<24}{'G':>4}   {header}")
    for model, g in (
        ("Llama-3-8B  32/8", 4),
        ("Llama-3-70B 64/8", 8),
        ("Mistral-7B  32/8", 4),
        ("DeepSeek MLA", 1),
    ):
        cells = []
        for L in (1, 2, 4, 8):
            ai = 4 * L * g / (2 * elem_bytes)
            cells.append(f"{ai:>5.0f}{'*' if ai >= ridge else ' '}")
        print(f"{model:<24}{g:>4}   " + "".join(f"{c:>7}" for c in cells))
    print()
    need = {g: ridge * (2 * elem_bytes) / (4 * g) for g in (1, 4, 8, 16)}
    print("跨过 ridge 所需的 query token 数 L（*=算力 bound）：")
    print("  " + "  ".join(f"G={g}: L>={L:.0f}" for g, L in need.items()))
    print()
    print("结论：投机解码把 L 从 1 抬到 γ+1，乘上 GQA 的复用，仍远低于 ridge；")
    print("      强度变高只是减少带宽余量，不改变「decode 是带宽 bound」这一性质。")
    print()


# ---------------------------------------------------------------------------
# 3. backend 对照表（源码快照）
# ---------------------------------------------------------------------------
SNAPSHOT = [
    # name, vendor, arch gate, 负责阶段, 布局, spec-as-decode
    ("trtllm-gen", "NVIDIA", "SM100/103/107",
     "decode+prefill", "强制 HND", "是（device 端裁 draft）"),
    ("XQA", "NVIDIA", "SM90 / SM120 家族",
     "decode (MQA/GQA, beam)", "HND 原生；TMA 支持 NHD", "是（SPEC_DEC 分支）"),
    ("FlashInfer native", "NVIDIA", "SM80+",
     "decode+prefill", "任意（stride-aware）", "否 → 退回 prefill 路径"),
    ("FA3 / FA4", "NVIDIA", "SM90 / SM100",
     "prefill 为主", "任意（stride-aware）", "不切分，varlen 一把跑"),
    ("aiter (AITER)", "AMD", "gfx9 CDNA3+",
     "decode (asm) + prefill", "LHBNC / SHUFFLE", "是（MTP kernel）"),
    ("hpc (hpc-ops)", "Tencent", "SM90 only",
     "decode(MTP)+prefill", "LBNHC = NHD", "是（mtp<=3）"),
    ("TurboQuant", "vLLM", "见 backend",
     "decode", "LBNHC = NHD", "否"),
]


def report_backends():
    print("=" * 78)
    print("[3] attention backend 对照（源码快照）")
    print("=" * 78)
    widths = (18, 9, 18, 22, 24, 24)
    header = ("内核", "厂商", "架构门槛", "负责阶段", "KV 布局", "spec-as-decode")
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print("-" * 78)
    for row in SNAPSHOT:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    print()
    print("读法：布局这一列和 spec 这一列不成正比。hpc 有 MTP kernel 却用 NHD，")
    print("      所以「spec decode ⇒ HND」是伪相关；真正决定布局的是 kernel 内层循环。")
    print()


# ---------------------------------------------------------------------------
# 4. --scan：从源码现场抽取 supported_kv_cache_layouts()
# ---------------------------------------------------------------------------
LAYOUT_RE = re.compile(r"KVCacheLayout\.([A-Z]+)")


def scan(root):
    backend_dir = os.path.join(root, "vllm", "v1", "attention", "backends")
    if not os.path.isdir(backend_dir):
        print(f"!! 找不到 {backend_dir}", file=sys.stderr)
        return 1
    print("=" * 78)
    print(f"[4] 现场扫描 {backend_dir}")
    print("=" * 78)
    hits = 0
    for dirpath, _, filenames in os.walk(backend_dir):
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            idx = src.find("def supported_kv_cache_layouts")
            if idx < 0:
                continue
            body = src[idx:idx + 1200]
            # 只保留到下一个同级 def 为止，避免把别处的方法算进来
            nxt = body.find("\n    def ", 10)
            if nxt > 0:
                body = body[:nxt]
            names = []
            for m in LAYOUT_RE.finditer(body):
                if m.group(1) not in names:
                    names.append(m.group(1))
            rel = os.path.relpath(path, backend_dir)
            layout_txt = ", ".join(names) if names else "(未声明 = 任意布局)"
            print(f"{rel:<52} {layout_txt}")
            hits += 1
    print()
    print(f"共 {hits} 个 backend 声明了布局偏好；未声明者读 stride，不读布局名。")
    print()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", metavar="VLLM_ROOT",
                    help="vLLM 仓库根目录，从源码现场抽取布局声明")
    ap.add_argument("--tflops", type=float, default=989.0, help="fp16 峰值 TFLOPS")
    ap.add_argument("--bw", type=float, default=3.35, help="HBM 带宽 TB/s")
    ap.add_argument("--gqa", type=int, default=8, help="GQA 组大小 G")
    ap.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS,
                    help="模拟的历史 page 数")
    args = ap.parse_args()

    report_layouts(DEFAULT_PAGE, ELEM_BYTES, args.blocks)
    report_roofline(args.tflops, args.bw, args.gqa, ELEM_BYTES)
    if args.scan:
        return scan(args.scan)
    report_backends()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

依赖：Python 3.10+，仅标准库，不需要 GPU。`--scan` 需要一份 vLLM 源码。
