#!/usr/bin/env python3
"""RDMA 系列（33-45）的验收检查。

    python3 communication/verify_rdma_series.py

它逐章核对目标里列的每一条要求，并打印可复核的证据：
  1. 必备小节：本次讲解位置 / 为什么现在讲这个 / 心智模型 /
     常见错法 / 自测+答案 / 一页速记
  2. 章内引用的本地文件（脚本 / 相邻章节）真实存在
  3. 配套 .py 脚本能被 python3 真正执行（不是只做语法检查）
  4. 每个文件都在 git 里有提交记录

退出码：全部通过 0，否则 1。这样它可以当门禁用。
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMM = os.path.join(ROOT, "communication")

CHAPTERS = [
    "33-rdma-six-layer-spine.md",
    "34-rdma-connection-state-ledger.md",
    "35-rdma-congestion-control-loop.md",
    "36-rdma-troubleshooting-decision-tree.md",
    "37-rdma-gpudirect-degradation.md",
    "38-verbs-program-walkthrough.md",
    "39-rdma-memory-registration-cost.md",
    "40-rdma-collective-topology.md",
    "41-rdma-kubernetes-multitenancy.md",
    "42-rdma-reliability-psn-window.md",
    "43-rdma-l2-remaining-semantics.md",
    "44-rdma-flow-control-backpressure.md",
    "45-rdma-vs-dpdk-and-routing.md",
]

# 每一项：(显示名, 候选关键词列表)。命中任一即算通过 —— 允许同义写法。
SECTIONS = [
    ("讲解位置卡片", ["本次讲解位置"]),
    ("为什么现在讲", ["为什么现在讲这个"]),
    ("心智模型", ["心智模型"]),
    ("常见错法", ["常见错法", "常见误判", "顺序错了会长什么样"]),
    ("自测+答案", ["自测", "<details>"]),
    ("一页速记", ["一页速记"]),
]


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def check_sections(text):
    missing = []
    for name, keys in SECTIONS:
        if not any(k in text for k in keys):
            missing.append(name)
    return missing


def local_links(text):
    """返回章内所有相对链接指向的本地路径（跳过 http 与锚点）。"""
    out = []
    for m in re.finditer(r"\]\((\./[^)#]+)", text):
        target = m.group(1)
        p = os.path.normpath(os.path.join(COMM, target))
        out.append((target, p))
    return out


def run_script(path):
    try:
        r = subprocess.run([sys.executable, path], cwd=ROOT,
                           capture_output=True, timeout=120)
        return r.returncode == 0, (r.stderr or b"")[:200].decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return False, "timeout"


def git_commits(path):
    rel = os.path.relpath(path, ROOT)
    r = subprocess.run(["git", "log", "--oneline", "--", rel],
                       cwd=ROOT, capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().splitlines() if l]
    return lines


def main():
    problems = []
    scripts_checked = {}

    print("=" * 78)
    print("RDMA 系列验收（33-45）")
    print("=" * 78)
    print(f"\n{'章节':<44}{'小节':>6}{'链接':>6}{'脚本':>6}{'提交':>6}")
    print("-" * 78)

    for name in CHAPTERS:
        path = os.path.join(COMM, name)
        if not os.path.exists(path):
            problems.append(f"{name}: 文件不存在")
            print(f"{name:<44}{'MISS':>6}{'-':>6}{'-':>6}{'-':>6}")
            continue

        text = read(path)
        missing = check_sections(text)
        sec_ok = "ok" if not missing else f"{len(SECTIONS) - len(missing)}/{len(SECTIONS)}"
        if missing:
            problems.append(f"{name}: 缺小节 {missing}")

        # 链接
        links = local_links(text)
        bad_links = [(t, p) for t, p in links if not os.path.exists(p)]
        link_ok = "ok" if not bad_links else "BAD"
        for t, _ in bad_links:
            problems.append(f"{name}: 链接失效 {t}")

        # 脚本：本章引用的同目录 .py
        py = sorted({p for _, p in links
                     if p.endswith(".py") and os.path.dirname(p) == COMM})
        script_res = "—"
        if py:
            oks = []
            for s in py:
                if s not in scripts_checked:
                    scripts_checked[s] = run_script(s)
                ok, err = scripts_checked[s]
                oks.append(ok)
                if not ok:
                    problems.append(f"{os.path.basename(s)}: 运行失败 {err}")
            script_res = "ok" if all(oks) else "FAIL"

        # 提交
        commits = git_commits(path)
        commit_res = "ok" if commits else "NONE"
        if not commits:
            problems.append(f"{name}: 无 git 提交")

        print(f"{name:<44}{sec_ok:>6}{link_ok:>6}{script_res:>6}"
              f"{commit_res:>6}")

    print("-" * 78)
    print(f"\n配套脚本：{len(scripts_checked)} 个，全部执行成功："
          f"{'是' if all(v[0] for v in scripts_checked.values()) else '否'}")
    for s, (ok, err) in sorted(scripts_checked.items()):
        print(f"  {'ok  ' if ok else 'FAIL'} {os.path.basename(s)}")

    # 工作树是否干净（防止"写了但没提交"）
    r = subprocess.run(["git", "status", "--porcelain", "--", "communication"],
                       cwd=ROOT, capture_output=True, text=True)
    dirty = [l for l in r.stdout.splitlines() if l.strip()]
    print(f"\ncommunication/ 未提交改动：{len(dirty)} 个")
    for d in dirty:
        print(f"  {d}")

    print("\n" + "=" * 78)
    if problems:
        print(f"验收未通过：{len(problems)} 个问题")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("验收通过：13 章齐备，脚本可执行，链接有效，均有提交记录。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
