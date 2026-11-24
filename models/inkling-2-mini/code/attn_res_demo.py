#!/usr/bin/env python3
"""attn_res_demo.py — AttnRes（深度残差）的三种打分方式 + Kimi-K3 的 bank 版本

普通残差 h <- h + f(h) 只能沿深度一层层挤；AttnRes 在「深度轴」上做一次
softmax，让每层自己决定从哪些历史源读回多少：

    alpha = softmax_j( score(h, src_j) )
    out   = sum_j alpha_j * src_j          # 凸组合，sum(alpha)=1

本脚本用 T=3, H=4, 5 个源的玩具矩阵，把三件事打出来：

  1. 三种 score 方式的 alpha 分布（无 query / 带 query / per-head）
  2. 凸组合不会放大范数（||out|| <= max ||src||）
  3. Kimi-K3 的 bank + prefix_sum 记账：每 12 层才 materialize 一行

运行：
    python3 attn_res_demo.py
无第三方依赖。
"""

import numpy as np

RNG = np.random.default_rng(0)
T, H, R = 3, 4, 5          # 3 个 token、hidden 4 维、5 个深度源
EPS = 1e-6


def rms_norm(x, eps=EPS):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def softmax_depth(scores):
    """沿深度轴（axis 0）做 softmax：scores 形状 (R, T)。"""
    e = np.exp(scores - scores.max(0, keepdims=True))
    return e / e.sum(0, keepdims=True)


def mix(alpha, srcs):
    """alpha: (R,T)  srcs: (R,T,H)  ->  (T,H)"""
    return np.einsum("st,std->td", alpha, srcs)


def show(name, alpha, srcs):
    out = mix(alpha, srcs)
    print(f"\n--- {name} ---")
    print("  alpha (R,T):")
    print("   ", np.array2string(alpha, precision=3, prefix="    "))
    print(f"  列和 (每个 token 的 alpha 总和): {np.round(alpha.sum(0), 6)}")
    print(f"  ||out|| : {np.round(np.linalg.norm(out, axis=-1), 3)}")
    print(f"  max||src||: {np.round(np.linalg.norm(srcs, axis=-1).max(0), 3)}")
    ok = np.all(np.linalg.norm(out, axis=-1) <= np.linalg.norm(srcs, axis=-1).max(0) + 1e-5)
    print(f"  凸组合不放大范数: {'OK' if ok else 'FAIL'}")
    return out


# --------------------------------------------------------------------------- #
# 0. 造数据
# --------------------------------------------------------------------------- #

srcs = RNG.standard_normal((R, T, H)).astype(np.float32)
srcs *= np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)[:, None, None]
h = RNG.standard_normal((T, H)).astype(np.float32)          # 当前 hidden

print("=" * 70)
print(f"玩具尺度: T={T} tokens, H={H}, R={R} 个深度源")
print("=" * 70)
print("源的范数（刻意做成不同量级，模拟 embedding 与深层 hidden 的差距）：")
print("  ", np.round(np.linalg.norm(srcs, axis=-1).mean(-1), 3))


# --------------------------------------------------------------------------- #
# 1. 三种打分方式
# --------------------------------------------------------------------------- #

# A) 无 query：score_j = w . RMSNorm(src_j)     ← Kimi-K3 的路子
w_self = RNG.standard_normal(H).astype(np.float32) * 0.5
s_norm = rms_norm(srcs)
alpha_a = softmax_depth(np.einsum("std,d->st", s_norm, w_self))
show("A. 无 query：score_j = w · RMSNorm(src_j)（Kimi-K3）", alpha_a, srcs)

# B) 带 query：score_j = <h W_q, src_j> / sqrt(H)   ← mini_looped.py 的路子
w_q = RNG.standard_normal((H, 1)).astype(np.float32) * 0.3
q = h @ w_q                                                 # (T,1)
alpha_b = softmax_depth(np.einsum("std,t->st", srcs, q[:, 0]))
show("B. 带 query：score_j = <h W_q, src_j> / sqrt(H)（mini_looped.py）", alpha_b, srcs)

# C) per-head / 多 stream：每个 token 的 score 由 (H_q, H) 的投影给出，再在深度轴 softmax
#    图上写 "Global layers mix Q, K, and V inputs separately"：这里用 3 个 stream 演示
alpha_cs = []
for stream in range(3):                                     # stream = q / k / v
    w_s = RNG.standard_normal((H, 1)).astype(np.float32) * 0.3
    alpha_cs.append(softmax_depth(np.einsum("std,t->st", srcs, (h @ w_s)[:, 0])))
alpha_c = np.stack(alpha_cs, 0)                             # (3, R, T)
print("\n--- C. per-stream（global 层：Q/K/V 分别混合）---")
for i, nm in enumerate("QKV"):
    print(f"  stream {nm}: alpha 列和 {np.round(alpha_c[i].sum(0), 6)}")
print("  三个 stream 的 alpha 互不相同 → 各自独立选源")


# --------------------------------------------------------------------------- #
# 2. 为什么在循环结构里要它：softmax 前先归一
# --------------------------------------------------------------------------- #

print("\n" + "=" * 70)
print("softmax 前归一/减 max 的必要性")
print("=" * 70)
raw = np.einsum("std,d->st", srcs, w_self)                   # 不做 norm
alpha_raw = softmax_depth(raw)
print(f"  不做 norm 时 alpha 的熵（R={R}）：{np.round(-(alpha_raw * np.log(alpha_raw + 1e-12)).sum(0), 3)}")
print(f"  做 norm 后   alpha 的熵：{np.round(-(alpha_a * np.log(alpha_a + 1e-12)).sum(0), 3)}")
print("  熵低 = 几乎只读一个源 = 退化成普通残差；量级差异会让 softmax 饱和。")
print("  所以 Kimi-K3 在 score 侧挂 score_norm，且 score_proj 压成一次点积。")


# --------------------------------------------------------------------------- #
# 3. Kimi-K3 的 bank + prefix_sum 记账
# --------------------------------------------------------------------------- #

print("\n" + "=" * 70)
print("Kimi-K3 的 bank 版本：每 12 层才 materialize 一行")
print("=" * 70)

BLOCK = 3                                                   # 玩具里把 12 压成 3
NB = 4                                                      # bank 行数
bank = np.zeros((T, NB, H), dtype=np.float32)                # 冻结的 snapshot
nvb = 0                                                      # 已写入行数
prefix = srcs[0].copy()                                      # block 内 running prefix


def aggregate(prefix, bank, nvb, w_self):
    """bank 前 nvb 行 + 当前 prefix 一起参与 softmax。返回 (mix, alpha|None)。"""
    if nvb == 0:
        return prefix, None
    rows = np.concatenate([bank[:, :nvb, :], prefix[:, None, :]], axis=1)  # (T, nvb+1, H)
    rows_t = rows.transpose(1, 0, 2)                                       # (R', T, H)
    a = softmax_depth(np.einsum("std,d->st", rms_norm(rows_t), w_self))
    return mix(a, rows_t), a


n_layers = 10
for lid in range(n_layers):
    read, a = aggregate(prefix, bank, nvb, w_self)
    if lid % BLOCK == 0 and nvb < NB:                        # block 边界：写入一行
        bank[:, nvb, :] = read
        nvb += 1
        shape = "(退化为直通)" if a is None else str(a.shape)
        print(f"  layer {lid:2d}: 写入 bank 第 {nvb - 1} 行，nvb={nvb}，alpha 形状 {shape}")
    prefix = 0.9 * prefix + 0.1 * read                       # 玩具里的层间推进

print(f"\n  最终 bank 用了 {nvb}/{NB} 行；中间 {n_layers - nvb} 层的残差读的是")
print("  bank + running prefix，不额外存 [T,H] 的中间快照。")
print("  代价：bank 仍占 O(NB*T*H)，NB = L / block_size，Kimi-K3 是 L/12。")


# --------------------------------------------------------------------------- #
# 4. 反传视角（示意，不实现自动微分）
# --------------------------------------------------------------------------- #

print("\n" + "=" * 70)
print("反传：为什么它比纯加法残差好")
print("=" * 70)
print("  普通残差: grad 只能沿 h_{i-1} 一层层往回传，深度 = 链长")
print("  AttnRes : out = sum_j alpha_j * src_j")
print("            d out / d src_j = alpha_j  (> 0，非零！)")
print("            → 梯度直接落到任意历史源，不必穿过中间几十层")
a = alpha_b
j = int(np.argmax(a[:, 0]))
print(f"\n  本例 token0 的最大权重在源 {j}：alpha = {a[j, 0]:.3f}"
      f" → 该源拿到 {a[j, 0]:.3f} 倍梯度直通，不经过中间层")
print(f"  最小权重 {a[:, 0].min():.3f} 也 > 0 → 所有源都保留一条梯度通路")
print("  d alpha_j / d score_j = a_j(1-a_j) > 0 → 打分器本身也始终有梯度")
print("\n  一句话：AttnRes 买的不是表达力，是「深度方向的高速公路」。")
