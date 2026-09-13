#!/usr/bin/env python3
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(ROOT, "output", "output_z.bin")
golden = os.path.join(ROOT, "output", "golden_z.bin")
if not os.path.isfile(out) or not os.path.isfile(golden):
    print("missing output_z.bin or golden_z.bin", file=sys.stderr)
    sys.exit(1)

a = np.fromfile(out, dtype=np.float16)
b = np.fromfile(golden, dtype=np.float16)
if a.shape != b.shape:
    print(f"shape mismatch {a.shape} vs {b.shape}", file=sys.stderr)
    sys.exit(2)
# half add may differ slightly on CPU twin vs golden float path
max_abs = float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
print(f"max_abs_diff={max_abs}")
sys.exit(0 if max_abs < 1e-2 else 3)
