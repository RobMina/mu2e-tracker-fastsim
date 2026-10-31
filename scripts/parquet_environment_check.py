"""Verify an additive Parquet environment without changing the underlying image."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import pyarrow


def check(expected_root=None):
    location = Path(pyarrow.__file__).resolve()
    if expected_root is not None and not location.is_relative_to(Path(expected_root).resolve()):
        raise ValueError(f"pyarrow imported outside the requested environment: {location}")
    frame = pd.DataFrame({
        "identity": pd.Series([1, 2, None], dtype="Int64"),
        "label": pd.Series(["muon", None, ""], dtype="string"),
        "energy": [1.25, np.nan, 0.0],
        "selected": pd.Series([True, False, None], dtype="boolean"),
    })
    for candidate in (frame, frame.iloc[:0]):
        stream = io.BytesIO()
        candidate.to_parquet(stream, engine="pyarrow", index=False)
        stream.seek(0)
        pd.testing.assert_frame_equal(candidate.reset_index(drop=True), pd.read_parquet(stream, engine="pyarrow"))
    packages = ("numpy", "pandas", "jax", "jaxlib", "scipy", "uproot", "awkward", "pytest", "pyarrow")
    return {"python": platform.python_version(), "executable": sys.executable,
            "packages": {name: importlib.metadata.version(name) for name in packages},
            "pyarrow_path": str(location),
            "pyarrow_init_sha256": hashlib.sha256(location.read_bytes()).hexdigest(),
            "roundtrip_nullable": True, "roundtrip_empty": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.expected_root)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
