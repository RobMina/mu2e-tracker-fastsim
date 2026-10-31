"""Read-only GDML inventory for actual expanded-census primitive/Boolean requirements."""
import argparse
import json
from pathlib import Path

from scripts.ordered_torus_pair_probe import load_pair_sources
from scripts.transport_capture_audit import digest
from mu2efastsim import world


def run(directory, gdml, source, out, *, overlap_placements=None):
    directory, gdml, source, out = map(Path, (directory, gdml, source, out))
    receipt = json.loads((directory / "complete.json").read_text())
    if overlap_placements is not None:
        chosen = {r[key] for r in receipt["source_pairs"] for key in ("source_a", "source_b")}
        reports = [r for r in json.loads(Path(overlap_placements).read_text()) if r["source"] in chosen]
        if {r["source"] for r in reports} != chosen:
            raise ValueError("overlap sources require complete placement reports")
    else:
        reports = receipt["unresolved_placement_reports"]
    modules = load_pair_sources(source)
    index = modules["omittedpath"].PlacementIndex(gdml, [r["source"] for r in reports])
    summaries = []
    def shape(element):
        result = dict(tag=element.tag, attributes=element.attrib)
        if element.tag in ("subtraction", "union", "intersection"):
            result["first"] = shape(index.solids[element.find("first").get("ref")])
            result["second"] = shape(index.solids[element.find("second").get("ref")])
            result["transform"] = [{"tag": c.tag, "attributes": c.attrib} for c in element if c.tag not in ("first", "second")]
        else:
            result["children"] = [{"tag": c.tag, "attributes": c.attrib} for c in element]
        return result
    for report in reports:
        placement = index.placements[report["source"]]
        solid = index.solids[placement["solid"]]
        pieces = world.signed_local(solid, index.solids)
        summaries.append(dict(source=report["source"], material=report["material"], parent_material=report["parent_material"],
            leaf=placement["is_leaf"], rotation=placement["rotation"].tolist(), position=placement["position"].tolist(),
            ancestors=[dict(path=a["path"], solid=a["solid"], material=a["material"]) for a in placement["ancestors"]],
            shape=shape(solid), signed_pieces=None if pieces is None else len(pieces)))
    result = dict(scope="read-only geometry requirements, not intersection/material admission", placements=summaries,
        input_sha256={str(gdml): digest(gdml), str(directory / "complete.json"): digest(directory / "complete.json")})
    if overlap_placements is not None:
        result["input_sha256"][str(overlap_placements)] = digest(overlap_placements)
    result["source_sha256"] = {str(Path(__file__)): digest(__file__)}
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    for name in ("gdml", "source", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--overlap-placements")
    args = parser.parse_args()
    run(args.directory, args.gdml, args.source, args.out, overlap_placements=args.overlap_placements)
