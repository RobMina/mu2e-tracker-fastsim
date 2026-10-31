"""Saved-only precision rescore of nested bank trials, never relabeling pilots as T-V3 errors."""
import argparse
import json
from pathlib import Path

from scripts.secondary_bank_replicas import variance_components
from scripts.transport_capture_audit import digest


def run(campaign, out):
    campaign, out = Path(campaign), Path(out)
    path = campaign / "complete.json"
    before = digest(path)
    saved = json.loads(path.read_text())
    rows = saved.get("replicas", saved.get("rows"))
    if not rows:
        raise ValueError("completed nested bank trial rows required")
    banks = sorted({r["bank"] for r in rows})
    draws = sorted({r["draw"] for r in rows})
    if len(rows) != len(banks) * len(draws) or len({(r["bank"], r["draw"]) for r in rows}) != len(rows):
        raise ValueError("incomplete or duplicate bank/draw rectangle")
    result = dict(scope=saved["scope"], input_sha256=before, variance={},
                  source_sha256={str(Path(__file__)): digest(__file__)})
    for name in rows[0]["metrics"]:
        values = [[next(r for r in rows if r["bank"] == b and r["draw"] == d)["metrics"][name] for d in draws] for b in banks]
        values = [[v["mean"] if isinstance(v, dict) else v for v in row] for row in values]
        result["variance"][name] = variance_components(values)
    if before != digest(path):
        raise ValueError("nested bank trial changed")
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.campaign, args.out)
