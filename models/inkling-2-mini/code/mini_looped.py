#!/usr/bin/env python3
"""mini_looped.py — looped transformer 骨架的最小可运行实现（numpy，无第三方依赖）

目的不是复现 Inkling 2 Mini，而是把这篇笔记里的三本账跑出来、看明白：

  1. distinct layers  : 有多少份独立权重
  2. applications     : 一个 token 前向走过多少次层
  3. weight touches   : 一个 token 实际读取了多少参数（looped 的代价在这里）

同时按 Inkling 2 Mini 图上给出的四个数字（2304 / 128 / top-8 / 1152）
做一遍参数账反推，和 26.4B / 3.4B 对照（见 architecture.md §6）。

运行：
    python3 mini_looped.py            # 玩具前向 + 记账
    python3 mini_looped.py --real     # 只用算术核对 26.4B / 3.4B / 5.4B

玩具配置刻意保持“骨架同构、尺寸缩小”：D=32、core 3 层 × 3 轮、4 专家 top-2。
"""

import argparse
import numpy as np

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


class Config:
    # 结构：prelude / core / coda，中间段跑 K 轮
    n_pre, n_core, n_coda, n_pass = 2, 3, 3, 3

    # 尺寸（玩具）
    d_model = 32
    vocab = 48
    seq = 8
    n_q_heads = 4
    n_kv_heads = 2          # GQA 2:1
    window = 4              # local 窗口（每 6 层有一个 global）
    local_global = 6        # 6 层里 1 层 global → 与真模型 5:1 同构
    conv_k = 4
    n_experts = 4
    top_k = 2
    d_ff = 16
    softcap = 30.0

    @property
    def head_dim(self):
        return self.d_model // self.n_q_heads


CFG = Config()

# --------------------------------------------------------------------------- #
# 基础层
# --------------------------------------------------------------------------- #


def rms_norm(x, w, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Params(dict):
    """平面参数容器：名字 -> ndarray。方便统计参数量。"""

    def add(self, name, shape, rng, scale=None):
        scale = scale if scale is not None else 1.0 / np.sqrt(shape[0])
        self[name] = (rng.standard_normal(shape) * scale).astype(np.float32)
        return self[name]

    def n_params(self):
        return sum(v.size for v in self.values())


def short_conv(x, w):
    """causal depthwise conv，核长 k：只做时间轴局部混合。"""
    k = w.shape[0]
    T = x.shape[0]
    pad = np.zeros((k - 1, x.shape[1]), dtype=x.dtype)
    xp = np.concatenate([pad, x], axis=0)
    out = np.zeros_like(x)
    for i in range(k):
        out += w[i] * xp[i:i + T]
    return out


def gqa(x, p, cfg, is_global):
    """GQA + causal mask + QK L2 clip + V norm + learned rel-pos bias。"""
    T, D = x.shape
    hd = cfg.head_dim
    grp = cfg.n_q_heads // cfg.n_kv_heads

    q = (x @ p["wq"]).reshape(T, cfg.n_q_heads, hd)
    k = (x @ p["wk"]).reshape(T, cfg.n_kv_heads, hd)
    v = (x @ p["wv"]).reshape(T, cfg.n_kv_heads, hd)

    # QK L2 clip：把每行范数裁到 <= 1，防止 logits 随轮次放大
    for t in (q, k):
        n = np.linalg.norm(t, axis=-1, keepdims=True) + 1e-8
        t *= np.minimum(1.0, 1.0 / n)
    # V norm：值域归一
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)

    idx = np.arange(T)
    dist = idx[:, None] - idx[None, :]                      # i - j
    mask = dist >= 0                                        # causal
    if not is_global:                                       # local 窗口
        mask &= dist < cfg.window

    # learned relative-position bias：每头一张 (2T-1) 的表，按 i-j 取
    bias = p["rel_pos"][:, dist + T - 1]                    # (H_Q, T, T)

    out = np.zeros((T, cfg.n_q_heads, hd), dtype=x.dtype)
    for h in range(cfg.n_q_heads):
        kh = h // grp
        s = (q[:, h] @ k[:, kh].T) / np.sqrt(hd) + bias[h]
        s = np.where(mask, s, -1e30)
        s = np.exp(s - s.max(-1, keepdims=True))
        s = s / s.sum(-1, keepdims=True)
        out[:, h] = s @ v[:, kh]
    return out.reshape(T, D) @ p["wo"]


def swiglu(x, p, tag):
    """专家内部：SwiGLU + gate softcap。"""
    g = x @ p[f"{tag}.wg"]
    u = x @ p[f"{tag}.wu"]
    c = CFG.softcap
    g = c * np.tanh(g / c)                                  # gate softcap
    return (g * sigmoid(g) * u) @ p[f"{tag}.wd"]


def moe(x, p, cfg):
    """router → top-k → 加权合并；返回 (out, active_param_fraction)。"""
    logits = x @ p["router"]                                # (T, E)
    logits -= logits.max(-1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(-1, keepdims=True)

    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        top = np.argsort(prob[t])[-cfg.top_k:]
        w = prob[t, top] / prob[t, top].sum()
        for e, wt in zip(top, w):
            out[t] += wt * swiglu(x[t], p, f"exp{e}")
    return out


def attn_res(h, srcs, p):
    """深度残差：在深度轴上 softmax 选择，从历史 block 读回信息。"""
    S = h @ p["ar_w"]                                       # (T, 1) 每个源的权重
    s = np.concatenate(srcs, axis=0).reshape(len(srcs), *h.shape)  # (S, T, D)
    scores = np.einsum("std,t->st", s, S[:, 0])             # (S, T)
    scores -= scores.max(0, keepdims=True)
    a = np.exp(scores)
    a /= a.sum(0, keepdims=True)
    return np.einsum("st,std->td", a, s)


# --------------------------------------------------------------------------- #
# 层
# --------------------------------------------------------------------------- #


def make_layer(rng, cfg, layer_id, with_moe=True):
    """一层 core：门控残差 + AttnRes + (conv→GQA) + (conv→MoE)。"""
    D, hd = cfg.d_model, cfg.head_dim
    p = Params()
    p.add("n1", (D,), rng, 1.0)
    p.add("n2", (D,), rng, 1.0)
    p.add("wq", (D, cfg.n_q_heads * hd), rng)
    p.add("wk", (D, cfg.n_kv_heads * hd), rng)
    p.add("wv", (D, cfg.n_kv_heads * hd), rng)
    p.add("wo", (D, D), rng)
    p.add("rel_pos", (cfg.n_q_heads, 2 * cfg.seq - 1), rng, 0.01)
    p.add("conv_a", (cfg.conv_k, D), rng, 0.1)
    p.add("g1", (D, D), rng)                                # 门控残差 W1
    p.add("msg", (2 * D, D), rng)                           # 跨 pass 门控 W_msg
    p.add("ar_w", (D, 1), rng, 0.1)                         # AttnRes 读权重
    if with_moe:
        p.add("conv_f", (cfg.conv_k, D), rng, 0.1)
        p.add("g2", (D, D), rng)
        p.add("router", (D, cfg.n_experts), rng)
        for e in range(cfg.n_experts):
            p.add(f"exp{e}.wg", (D, cfg.d_ff), rng)
            p.add(f"exp{e}.wu", (D, cfg.d_ff), rng)
            p.add(f"exp{e}.wd", (cfg.d_ff, D), rng)
    return p


def run_layer(h, msg, srcs, p, cfg, is_global, calls):
    calls.append(1)

    # ① 跨 pass 门控注入（p >= 1 才有；looped 特有）
    if msg is not None:
        g = sigmoid(np.concatenate([h, msg], -1) @ p["msg"])
        h = h + g * msg

    # ② 深度残差
    h = h + attn_res(h, srcs, p)

    # ③ 注意力子层：conv → GQA → 门控残差
    u = gqa(short_conv(rms_norm(h, p["n1"]), p["conv_a"]), p, cfg, is_global)
    h = h + sigmoid(h @ p["g1"]) * u

    # ④ MoE 子层：conv → top-k 专家 → 门控残差
    if "router" in p:
        v = moe(short_conv(rms_norm(h, p["n2"]), p["conv_f"]), p, cfg)
        h = h + sigmoid(h @ p["g2"]) * v
    return h


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #


def build(cfg):
    rng = np.random.default_rng(0)
    layers = {}                                             # (阶段, 层号) -> Params
    for i in range(cfg.n_pre):
        layers[("pre", i)] = make_layer(rng, cfg, i)
    for i in range(cfg.n_core):
        layers[("core", i)] = make_layer(rng, cfg, cfg.n_pre + i)
    for i in range(cfg.n_coda):
        layers[("coda", i)] = make_layer(rng, cfg, cfg.n_pre + cfg.n_core + i)
    emb = rng.standard_normal((cfg.vocab, cfg.d_model)).astype(np.float32) * 0.02
    head = rng.standard_normal((cfg.d_model, cfg.vocab)).astype(np.float32) * 0.02
    norm = np.ones(cfg.d_model, dtype=np.float32)
    final_ar = rng.standard_normal((cfg.d_model, 1)).astype(np.float32) * 0.1
    return layers, emb, head, norm, final_ar


def forward(tok, cfg, layers, emb, head, norm, final_ar):
    calls = []
    x = emb[tok].copy()
    srcs = [x]

    for i in range(cfg.n_pre):                              # prelude：跑一次
        x = run_layer(x, None, srcs, layers[("pre", i)], cfg,
                      is_global=((i % cfg.local_global) == cfg.local_global - 1),
                      calls=calls)
        srcs.append(x)
    h0 = x                                                  # 三个 pass 同一起点

    prev = [None] * cfg.n_core
    for p_pass in range(cfg.n_pass):                        # core：同一批权重 × K
        h = h0
        for i in range(cfg.n_core):
            lid = cfg.n_pre + i
            h = run_layer(h, prev[i], srcs, layers[("core", i)], cfg,
                          is_global=((lid % cfg.local_global) == cfg.local_global - 1),
                          calls=calls)
            prev[i] = h
            srcs.append(h)

    for i in range(cfg.n_coda):                             # coda：跑一次
        lid = cfg.n_pre + cfg.n_core + i
        x = run_layer(h, None, srcs, layers[("coda", i)], cfg,
                      is_global=((lid % cfg.local_global) == cfg.local_global - 1),
                      calls=calls)
        srcs.append(x)

    x = x + attn_res(x, srcs, {"ar_w": final_ar})           # Final Block AttnRes
    x = rms_norm(x, norm)
    logits = x @ head
    c = cfg.softcap
    return c * np.tanh(logits / c), calls


# --------------------------------------------------------------------------- #
# 记账
# --------------------------------------------------------------------------- #


def active_share(cfg):
    """单层里"这个 token 实际用到"的权重占比（MoE 稀疏带来的折损）。"""
    D, F = cfg.d_model, cfg.d_ff
    attn = D * (cfg.n_q_heads * (D // cfg.n_q_heads)) \
        + 2 * D * (cfg.n_kv_heads * (D // cfg.n_q_heads)) \
        + D * D + 2 * D * D                                  # q,k,v,o + g1,msg
    moe_all = cfg.n_experts * 3 * D * F + D * cfg.n_experts
    moe_act = cfg.top_k * 3 * D * F + D * cfg.n_experts
    total = attn + moe_all
    return (attn + moe_act) / total, attn, moe_all


def report_toy(cfg, layers, emb, head, logits, calls):
    n_distinct = len(layers)
    n_apps = len(calls)
    total = sum(p.n_params() for p in layers.values()) + emb.size + head.size
    share, _, _ = active_share(cfg)
    active = total * 0                                     # 下面逐层算
    for key, p in layers.items():
        if "router" in p:
            active += p.n_params() * share
        else:
            active += p.n_params()
    active += emb.size + head.size

    print("=" * 70)
    print("玩具模型（骨架同构，尺寸缩小）")
    print("=" * 70)
    print(f"  结构              : prelude {cfg.n_pre} + core {cfg.n_core}×{cfg.n_pass}"
          f" + coda {cfg.n_coda}")
    print(f"  distinct layers   : {n_distinct}  (权重份数)")
    print(f"  applications      : {n_apps}  (一个 token 走过的层数)")
    print(f"  核对              : {cfg.n_pre} + {cfg.n_core}×{cfg.n_pass} + "
          f"{cfg.n_coda} = {cfg.n_pre + cfg.n_core * cfg.n_pass + cfg.n_coda}"
          f"  -> {'OK' if n_apps == cfg.n_pre + cfg.n_core * cfg.n_pass + cfg.n_coda else 'MISMATCH'}")
    assert n_apps == cfg.n_pre + cfg.n_core * cfg.n_pass + cfg.n_coda
    print(f"  distinct params   : {total:,}")
    print(f"  active params/token: {active:,.0f}  "
          f"({active / total:.0%} of distinct，MoE top-{cfg.top_k}/{cfg.n_experts})")
    print(f"  weight touches    : {active * n_apps / n_distinct:,.0f}  "
          f"(≈ active × {n_apps / n_distinct:.2f})")
    print(f"  logits            : shape={logits.shape}  "
          f"range=[{logits.min():.3f}, {logits.max():.3f}]  "
          f"anynan={np.isnan(logits).any()}")
    print()


def report_real(cfg_d=2304, n_experts=128, top_k=8, d_ff=1152,
                n_layers=24, n_core=12, n_pass=3, n_pre=6, n_coda=6,
                reported=(26.4e9, 3.4e9, 5.4e9)):
    """只用算术，核对 architecture.md §6 的反推。"""
    exp_per = 3 * cfg_d * d_ff                                  # SwiGLU 三矩阵
    moe_layer = n_experts * exp_per
    moe_total = n_layers * moe_layer
    dense = reported[0] - moe_total
    active = dense + n_layers * (top_k / n_experts) * moe_layer
    n_apps = n_pre + n_core * n_pass + n_coda
    exp_per_app = (top_k / n_experts) * moe_layer               # 每次应用读到的专家权重
    dense_per_app = dense / n_layers

    # 两种记账口径（图上没说是哪一种）
    a_only_experts = dense + n_apps * exp_per_app               # 稠密只计一次
    b_full = n_apps * (exp_per_app + dense_per_app)             # 稠密也按应用次数计

    print("=" * 70)
    print("Inkling 2 Mini 参数账反推（纯算术，对照 architecture.md §6）")
    print("=" * 70)
    print(f"  单专家 (3×{cfg_d}×{d_ff})            : {exp_per / 1e6:.2f} M")
    print(f"  {n_experts} 专家/层                 : {moe_layer / 1e9:.3f} B")
    print(f"  {n_layers} 层 MoE（假设铺满）        : {moe_total / 1e9:.2f} B"
          f"  ({moe_total / reported[0]:.0%} of total)")
    print(f"  稠密部分（反解）          : {dense / 1e9:.2f} B")
    print(f"  活跃参数/token            : {active / 1e9:.2f} B   "
          f"(图: {reported[1] / 1e9:.1f} B, 误差 {abs(active - reported[1]) / reported[1]:.1%})")
    print(f"  weight touches 口径 A     : {a_only_experts / 1e9:.2f} B   "
          f"(稠密只计一次 + 专家按应用次数)")
    print(f"  weight touches 口径 B     : {b_full / 1e9:.2f} B   "
          f"(稠密也按应用次数)")
    print(f"  图上的 weight activations : {reported[2] / 1e9:.1f} B")
    print(f"  层应用                    : {n_pre} + {n_core}×{n_pass} + {n_coda} = {n_apps}")
    print()
    print("  读法：26.4B 与 3.4B 能被同一套分解同时对上（<2%），这确定了")
    print("        「MoE 铺满 24 层、单层 128×7.96M」；而 5.4B 落在两种口径之间，")
    print("        所以只能拿到定性结论：多出来的部分来自中间 12 层被重复读。")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="只跑真模型的参数账反推")
    args = ap.parse_args()

    report_real()
    if args.real:
        return

    cfg = CFG
    layers, emb, head, norm, final_ar = build(cfg)
    rng = np.random.default_rng(1)
    tok = rng.integers(0, cfg.vocab, size=cfg.seq)
    logits, calls = forward(tok, cfg, layers, emb, head, norm, final_ar)
    report_toy(cfg, layers, emb, head, logits, calls)

    # 关键对照：把 K 设为 1，算力账立刻掉到普通模型的水平
    cfg2 = Config()
    cfg2.n_pass = 1
    layers2, emb2, head2, norm2, ar2 = build(cfg2)
    _, calls1 = forward(tok, cfg2, layers2, emb2, head2, norm2, ar2)
    print(f"  对照：K=1 时 applications = {len(calls1)}，K={cfg.n_pass} 时 = {len(calls)}；"
          f"权重的份数两者都是 {len(layers)} 份。")
    print("  这就是 looped 的全部交易：参数不变，前向变长。")


if __name__ == "__main__":
    main()
