"""Compare the CE chain's represented geometry with the resolved stock Run 1A geometry.

This checks effective SimpleConfig parameters and cechain's represented geometry only, not GDML navigation or
unrepresented supports. Run on captured runtime geometry sources, not a moving geom_common alias or laptop Offline.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

PREFIXES = ("stoppingTarget.", "protonabsorber.", "tsda.")
EXACT = ("hasTarget", "hasProtonAbsorber", "hasTSdA", "mu2e.solenoidOffset", "mu2e.detectorSystemZ0")


def differences(a, b):
    """JSON-ready exact differences without rounding or fitting geometry values."""
    return {k: {"chain": a.get(k), "stock": b.get(k)} for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def inspect(root, name):
    """Read-only beam geometry interface; hash all resolved include files for reproducibility."""
    from beamfastsim import cechain, simpleconfig
    files = []
    root = Path(root)
    path = root / name
    conf = simpleconfig.read(str(path), str(root), source_files=files)
    selected = {k: v for k, v in conf.items() if k.startswith(PREFIXES) or k in EXACT}
    source = [{"path": str(Path(f).relative_to(root)), "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest()}
              for f in files]
    represented = plain(asdict(cechain.from_offline(name, str(root))))
    return {"top": name, "effective_parameters": selected, "represented": represented, "source_files": source}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain-root", required=True)
    ap.add_argument("--stock-root", required=True)
    ap.add_argument("--chain-geom", default="Offline/Mu2eG4/geom/geom_2021_PhaseI_v03.txt")
    ap.add_argument("--stock-geom", default="Offline/Mu2eG4/geom/geom_run1_a.txt")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    chain, stock = inspect(a.chain_root, a.chain_geom), inspect(a.stock_root, a.stock_geom)
    report = {"chain": chain, "stock": stock,
              "effective_parameter_differences": differences(chain["effective_parameters"], stock["effective_parameters"]),
              "represented_geometry_differences": differences(chain["represented"], stock["represented"]),
              "scope": "represented cechain geometry and selected effective config only; not support/GDML completeness"}
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cegeometry.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
