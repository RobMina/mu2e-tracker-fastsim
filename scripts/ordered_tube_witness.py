"""Saved tube-source provenance and GDML requirements; no hierarchy repair or transport admission."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from mu2efastsim import world
from scripts.transport_capture_audit import digest
from trkfastsim import steps


def run(directory, gdml, out):
    directory, gdml, out = map(Path, (directory, gdml, out))
    paths = [directory / "rays.npz", directory / "adjusted_hits.csv", gdml]
    hashes = {str(p): digest(p) for p in paths}
    with np.load(paths[0], allow_pickle=False) as saved:
        bbox = saved["source_selection_bbox_mm"]
    labels = []
    tubes = world.load_tubes(gdml, bbox=bbox, skip_names=steps.INTERIOR_SKIP, names=labels)
    selected = {i: (tube, label) for i, (tube, label) in enumerate(zip(tubes, labels)) if tube[0] != "StrawGas"}
    raw = pd.read_csv(paths[1], float_precision="round_trip")
    sources = sorted(set(raw.loc[raw.source.str.startswith("tube:"), "source"]))
    root = ET.parse(gdml).getroot()
    volumes = {v.get("name"): v for v in root.find("structure").findall("volume")}
    solids = {s.get("name"): s for s in root.find("solids")}
    records = []
    for source in sources:
        code = int(source.removeprefix("tube:"))
        tube, label = selected[code]
        matches = [v for name, v in volumes.items() if world._strip(name) == label.rsplit("/", 1)[-1]]
        shapes = []
        for volume in matches:
            shape = solids[volume.find("solidref").get("ref")]
            children = []
            for pv in volume.findall("physvol"):
                child = volumes[pv.find("volumeref").get("ref")]
                solid = solids[child.find("solidref").get("ref")]
                children.append(dict(placement=pv.attrib, transform=[dict(tag=c.tag, attributes=c.attrib) for c in pv if c.tag != "volumeref"],
                    volume=child.get("name"), material=world._strip(child.find("materialref").get("ref")), solid=ET.tostring(solid, encoding="unicode")))
            shapes.append(dict(volume=volume.get("name"), solid=ET.tostring(shape, encoding="unicode"), children=children))
        records.append(dict(source=source, material=tube[0], logical_path=label, rotation=tube[1].tolist(), position=tube[2].tolist(),
            rmin_mm=tube[3], rmax_mm=tube[4], half_z_mm=tube[5], phi0_rad=tube[6], dphi_rad=tube[7], logical_matches=shapes))
    result = dict(scope="tube approximation provenance/requirements only, not authenticated physical ownership", sources=records,
        input_sha256=hashes, source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_physics=False)
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("tube witness inputs changed")
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(sources=len(records), out=str(out))))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.directory, args.gdml, args.out)
