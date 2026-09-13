#!/usr/bin/env python3
"""Generate half-precision-ish float16 bins for add_custom demo."""
import os
import struct
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR = os.path.join(ROOT, "input")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(IN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

n = 8 * 2048
rng = np.random.default_rng(0)
x = rng.standard_normal(n).astype(np.float16)
y = rng.standard_normal(n).astype(np.float16)
z = (x.astype(np.float32) + y.astype(np.float32)).astype(np.float16)

x.tofile(os.path.join(IN_DIR, "input_x.bin"))
y.tofile(os.path.join(IN_DIR, "input_y.bin"))
z.tofile(os.path.join(OUT_DIR, "golden_z.bin"))
print(f"wrote {n} f16 elems to {IN_DIR} and golden to {OUT_DIR}")
