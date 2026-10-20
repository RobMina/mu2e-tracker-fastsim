"""KPP tracker straws from the Geant4 GDML export of the extracted world (the same file as the CRV V0 check).

Placement chain (one StrawPanelEnvelope logical volume holds the 96 straw-gas tubes of a panel; it is placed once in
each of the 216 Panel<plane>_<panel> volumes):
    TrackerStrawGas_p_q_s  in StrawPanelEnvelope   (pos, rot: the straw axis is the tube's local z)
    StrawPanelEnvelope     in Panel<pl>_<pa>       (identity)
    Panel<pl>_<pa>         in TrackerPlaneEnvelope_<pl>
    TrackerPlaneEnvelope   in TrackerMother -> garageFakeDS3Vacuum -> HallAir -> World
StrawId follows StrawSD: straw = copy number in the panel envelope, panel/plane from the Panel volume name. The
sensitive gas is the TrackerStrawGas tube minus its daughter TrackerStrawWall (Mylar, rmin 2.5085 = the straw inner
radius): `radius` is that inner radius, read from the wall solid; half lengths come from the gas tube solids.
(T-V0, 2026-10-16: every G4 primary-muon step enters and leaves at 2.5085 mm.) Rotations use mu2efastsim.world._rot (the convention validated by the CRV V0 check).

Frames: the GDML world frame is the Mu2e frame for this world (CRV V0: Mu2e origin at (0,0,0)). StrawGasStep and
StepPointMC positions are in the DETECTOR frame (Mu2eG4 StrawSD subtracts the DetectorSystem origin, which for the
extracted geometry is (-mu2e.solenoidOffset, 0, mu2e.detectorSystemZ0) = (-3904, 0, 24171) mm). `to_detector` /
`to_mu2e` convert; the fast sim transports in the Mu2e frame (CRY primaries) and writes steps in the detector frame.

Alignment: `apply_alignment` moves the nominal straws with tracker, plane and panel rigid transforms in the
AlignedTrackerMaker composition (aligned = T_trk * (plane_to_DS * T_plane) * (panel_to_plane * T_panel)), so sim and
reco alignment are separate inputs.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace

import numpy as np

from mu2efastsim.world import _rot

NPLANES, NPANELS, NSTRAWS = 36, 6, 96
# DetectorSystem origin in the Mu2e frame, extracted geometry (geom_run1_a_stickman.txt solenoidOffset 3904;
# geom_common_extracted_v04.txt detectorSystemZ0 24171)
DETECTOR_ORIGIN_EXTRACTED = np.array([-3904.0, 0.0, 24171.0])


def to_detector(x, origin=DETECTOR_ORIGIN_EXTRACTED):
    return np.asarray(x, float) - origin


def to_mu2e(x, origin=DETECTOR_ORIGIN_EXTRACTED):
    return np.asarray(x, float) + origin
_PANEL = re.compile(r"^Panel(\d+)_(\d+)$")
_GAS = re.compile(r"^TrackerStrawGas_(\d+)_(\d+)_(\d+)$")
_STRIP = re.compile(r"0x[0-9a-f]+$")


def _strip(n):
    return _STRIP.sub("", n)


def straw_id(plane, panel, straw):
    """StrawId::asUint16: plane << 10 | panel << 7 | straw."""
    return (np.asarray(plane) << 10) | (np.asarray(panel) << 7) | np.asarray(straw)


@dataclass
class StrawGeometry:
    """All straws, index = uniqueStraw = (plane * 6 + panel) * 96 + straw. Positions in mm, Mu2e (G4 world) frame."""
    plane: np.ndarray
    panel: np.ndarray
    straw: np.ndarray
    mid: np.ndarray          # [N, 3] straw axis midpoint
    dir: np.ndarray          # [N, 3] unit axis
    halflen: np.ndarray      # [N]
    radius: float            # sensitive-gas radius = straw inner radius (wall rmin)
    # per panel (index = uniquePanel), for alignment: panel and plane frames in the world
    panel_R: np.ndarray      # [216, 3, 3] panel-envelope frame -> world rotation
    panel_t: np.ndarray      # [216, 3]
    plane_R: np.ndarray      # [36, 3, 3]
    plane_t: np.ndarray      # [36, 3]
    wall_rmax: float = None  # outer radius of the straw wall (Mylar + outer metal); None = radius + 0.015
    metal_t: float = 0.0     # thickness of the wall's outer metal layer (Al6061), inside wall_rmax

    @property
    def nstraws(self):
        return len(self.mid)

    @property
    def sid(self):
        return straw_id(self.plane, self.panel, self.straw)

    @property
    def upanel(self):
        return self.plane * NPANELS + self.panel


def _placement(pv):
    p = pv.find("position")
    pos = np.zeros(3) if p is None else np.array([float(p.get(a, 0)) for a in "xyz"])
    if p is not None and p.get("unit", "mm") != "mm":
        pos *= {"cm": 10.0, "m": 1000.0}[p.get("unit")]
    return _rot(pv.find("rotation")), pos


def from_gdml(path: str) -> StrawGeometry:
    root = ET.parse(path).getroot()
    vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
    solids = {s.get("name"): s for s in root.find("solids")}
    world = root.find("setup/world").get("ref")
    parent = {}
    for n, v in vols.items():
        for pv in v.findall("physvol"):
            parent.setdefault(pv.find("volumeref").get("ref"), []).append((n, pv))

    def to_world(lv):
        """(R, t) of a singly placed logical volume's frame in the world."""
        R, t = np.eye(3), np.zeros(3)
        while lv != world:
            (mother, pv), = parent[lv]
            Rl, tl = _placement(pv)
            R, t = Rl @ R, Rl @ t + tl
            lv = mother
        return R, t

    # straws in the (single) panel envelope
    env = [n for n in vols if _strip(n) == "StrawPanelEnvelope"]
    if len(env) != 1:
        raise ValueError(f"expected one StrawPanelEnvelope logical volume, found {len(env)}")
    env = env[0]
    s_loc = {}
    radius, wall_rmax, metal_t = None, None, 0.0
    for pv in vols[env].findall("physvol"):
        lv = pv.find("volumeref").get("ref")
        if not _GAS.match(_strip(lv)):
            continue
        copy = int(pv.get("copynumber", 0))
        R, t = _placement(pv)
        tube = solids[vols[lv].find("solidref").get("ref")]
        for d in vols[lv].findall("physvol"):
            dlv = d.find("volumeref").get("ref")
            if _strip(dlv).startswith("TrackerStrawWall_"):
                wsol = solids[vols[dlv].find("solidref").get("ref")]
                radius, wall_rmax = float(wsol.get("rmin")), float(wsol.get("rmax"))
                for m in vols[dlv].findall("physvol"):
                    mlv = m.find("volumeref").get("ref")
                    if _strip(mlv).startswith("TrackerStrawWallOuterMetal_"):
                        msol = solids[vols[mlv].find("solidref").get("ref")]
                        metal_t = float(msol.get("rmax")) - float(msol.get("rmin"))
        s_loc[copy] = (t, R @ np.array([0.0, 0.0, 1.0]), 0.5 * float(tube.get("z")))
    if sorted(s_loc) != list(range(NSTRAWS)):
        raise ValueError(f"panel envelope holds straw copies {sorted(s_loc)[:5]}..., expected 0..{NSTRAWS - 1}")

    # panel envelope frames in the world, via each Panel volume
    panel_R = np.zeros((NPLANES * NPANELS, 3, 3))
    panel_t = np.zeros((NPLANES * NPANELS, 3))
    plane_R = np.zeros((NPLANES, 3, 3))
    plane_t = np.zeros((NPLANES, 3))
    seen = 0
    for mother, pv in parent[env]:
        m = _PANEL.match(_strip(mother))
        if not m:
            continue
        pl, pa = map(int, m.groups())
        Rm, tm = to_world(mother)
        Rl, tl = _placement(pv)
        up = pl * NPANELS + pa
        panel_R[up], panel_t[up] = Rm @ Rl, Rm @ tl + tm
        (pmother, _), = parent[mother]
        plane_R[pl], plane_t[pl] = to_world(pmother)
        seen += 1
    if seen != NPLANES * NPANELS:
        raise ValueError(f"found {seen} panel placements, expected {NPLANES * NPANELS}")

    sl_t = np.array([s_loc[i][0] for i in range(NSTRAWS)])
    sl_d = np.array([s_loc[i][1] for i in range(NSTRAWS)])
    sl_h = np.array([s_loc[i][2] for i in range(NSTRAWS)])
    mid = (np.einsum("pij,sj->psi", panel_R, sl_t) + panel_t[:, None, :]).reshape(-1, 3)
    dirs = np.einsum("pij,sj->psi", panel_R, sl_d).reshape(-1, 3)
    up = np.repeat(np.arange(NPLANES * NPANELS), NSTRAWS)
    return StrawGeometry(plane=up // NPANELS, panel=up % NPANELS, straw=np.tile(np.arange(NSTRAWS), NPLANES * NPANELS),
                         mid=mid, dir=dirs / np.linalg.norm(dirs, axis=1)[:, None],
                         halflen=np.tile(sl_h, NPLANES * NPANELS), radius=radius,
                         panel_R=panel_R, panel_t=panel_t, plane_R=plane_R, plane_t=plane_t,
                         wall_rmax=wall_rmax, metal_t=metal_t)


def rotation_xyz(rx, ry, rz):
    """HepTransform rotation convention of TrkAlignParams: R = Rz(rz) Ry(ry) Rx(rx) (active), angles in rad."""
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def apply_alignment(g: StrawGeometry, tracker=None, planes=None, panels=None, tracker_origin=None) -> StrawGeometry:
    """Rigidly move straws, in the AlignedTrackerMaker composition, in the tracker (detector) frame.

    tracker: (dx, dy, dz, rx, ry, rz); planes: {plane: 6-tuple}; panels: {uniquePanel: 6-tuple}; mm and rad, the
    TrkAlign* table convention. The plane and panel transforms act in their own frames (plane_to_DS * T_plane), the
    tracker transform in the detector frame about tracker_origin (world position of the detector-frame origin).
    Straw-end (wire) offsets are not modelled here.
    """
    if not (tracker or planes or panels):
        return g
    o = np.zeros(3) if tracker_origin is None else np.asarray(tracker_origin, float)
    mid, dirs = g.mid.copy(), g.dir.copy()
    up = g.upanel

    def rigid(R, t, idx, Rf, tf):
        # transform acting in a local frame (Rf, tf): x -> Rf (R (Rf^T (x - tf)) + t) + tf
        loc = np.einsum("ij,nj->ni", Rf.T, mid[idx] - tf)
        mid[idx] = np.einsum("ij,nj->ni", Rf, loc @ R.T + t) + tf
        dirs[idx] = np.einsum("ij,nj->ni", Rf @ R @ Rf.T, dirs[idx])

    for p, a in (panels or {}).items():
        rigid(rotation_xyz(*a[3:]), np.asarray(a[:3], float), up == p, g.panel_R[p], g.panel_t[p])
    for p, a in (planes or {}).items():
        rigid(rotation_xyz(*a[3:]), np.asarray(a[:3], float), g.plane == p, g.plane_R[p], g.plane_t[p])
    if tracker:
        rigid(rotation_xyz(*tracker[3:]), np.asarray(tracker[:3], float), np.ones(len(mid), bool), np.eye(3), o)
    return replace(g, mid=mid, dir=dirs)
