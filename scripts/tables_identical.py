"""Check that two fast step-table files hold identical trees (every column of sgsteps, surfsteps, particles).

    python3 tracker/scripts/tables_identical.py A.root B.root
"""
import sys

import numpy as np
import uproot

a, b = (uproot.open(p) for p in sys.argv[1:3])
ok = True
for t in ("sgsteps", "surfsteps", "particles"):
    x, y = a[f"dump/{t}"].arrays(library="np"), b[f"dump/{t}"].arrays(library="np")
    same = x.keys() == y.keys() and all(np.array_equal(x[k], y[k]) for k in x)
    ok &= same
    print(f"{t}: {'identical' if same else 'DIFFERENT'} ({len(next(iter(x.values())))} vs {len(next(iter(y.values())))} rows)")
print("IDENTICAL" if ok else "DIFFERENT")
