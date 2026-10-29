"""Prepare isolated passive crossing capture sources; never build, install or run native physics."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path("/exp/mu2e/app/users/ram2aq")
BASE = ROOT/"fastreco_operands_p109_20261028"
AREA = ROOT/"fastreco_crossings_p109_20261028"
TEMPLATES = Path(__file__).resolve().parents[1]/"art/crossing_capture"
PINS = {"Offline": "9b3d463371f34731f8515d1a63914ccf070536a8",
        "KinKal": "bb23c26348fc6e5fcbeefb5c7138919ef21f7abc",
        "Production": "4479f3fad37a85ecd9485394198658237961dcc3",
        "mu2e-trig-config": "d3ec315c7e8a626e8cf9c417ab6522a8e9a8f669",
        "EventNtuple": "76ecb8af80c60e98d80a62a29f0082f36c20add5"}
FROZEN = {
    "KinKal/Fit/Track.hh": "3bff114b98002b835fccf4b564ce2e5d9e2629c844c5e0f0deb0a2ec2315150f",
    "KinKal/Fit/IterationTrace.hh": "48e77629a8f19eca8af65d23133e99e69830041adb028c21601d5962c766f364",
    "KinKal/Trajectory/PiecewiseClosestApproach.hh": "fda58b1eca579a19f7f4f786fc46caf1673c7f0d811377d6307b16705f516315",
    "Offline/Mu2eKinKal/inc/KKFit.hh": "d177547b7cb72b03f2729595f0b24a8f2aba52c47b0f597581cd0da4d232720f",
    "Offline/Mu2eKinKal/inc/KKStrawXing.hh": "bd628dc818cc11c1a02fa7c4bbe02fdb48cbd738fb945365bf127b8785b73e59",
    "TrkFastSim/src/KinematicLineFitTrace_module.cc": "4707dbcd42484e23ffcc2bfa8fecd9c3dfefb9eb2113c5714951d4a6e22409f3",
    "TrkFastSim/inc/IterationCaptureWriter.hh": "6cff60457bd660d8c93256e2e47556af5d0dc8097be7e78f2ed6158b63cd26dd",
    "TrkFastSim/inc/HitOperandCapture.hh": "2f6f341ba1246c0b285d92cbf0a90a11f12e0d7747ccb1ed0d09fe78fcac34d6",
    "Offline/TrackerGeom/inc/Tracker.hh": "fcef416f053345b5d57905ef33f065102a5274c43fc3d6bb1978539ad40fcbdb",
    "Offline/TrackerGeom/inc/Plane.hh": "2f9f3baa5ab3b2a89a2f765e632997d99194fcc4110a6f0504ae15bf6d5321f2",
    "Offline/TrackerGeom/inc/Panel.hh": "54fafa34a1c65c5d901dc81a7ddd4901b98ed74bc313c6c60abdab0ef74ee9c3",
    "Offline/TrackerGeom/inc/Straw.hh": "a651a9282959599c0bd7e77441aeed19e2e0ed4a5f311cb184b174f739075fe3",
    "Offline/GeneralUtilities/inc/HepTransform.hh": "14b957d1208bdf80df1a732e50e1adeba991f6ad389f6321c7e778e6a0bfbc9b",
    "Offline/GeneralUtilities/src/HepTransform.cc": "2968a281dd4f5e2a12e59bfec007f689887078058750ef42eee517110f9444c0",
    "Offline/DataProducts/inc/StrawId.hh": "ee28da49bd5b69a65ff13f62ff667ead445842e779c0f8484b679013162db50b",
    "KinKal/Trajectory/SensorLine.cc": "08ccaec6ead467767b5e6747eb1333d6b41d5fa4beeca859ce46aee0c3c1babf",
    "KinKal/Trajectory/ClosestApproach.hh": "ba54d8d91036d0085fed44ce432ff8b1926352ed8a88287b476f8eb4706d66d1",
    "KinKal/Trajectory/ClosestApproachData.hh": "d8459b962aea78eead06f72070952d7c3c1a8c71ec0ebdf37643976b2e939360",
    "Offline/TrackerGeom/src/Tracker.cc": "28215c3118f94b348afb0da7013a309540aa31ff26bfe8fbb4b9bf1dd24eaf77",
    "Offline/Mu2eKinKal/src/KKFitUtilities.cc": "9ac76a6219bf310a7889bc6a6926e8f7a1f44d0841d841b18d0770d0ea8a607e",
    "Offline/Mu2eKinKal/inc/KKTrack.hh": "449fc2ccd41a4eb7df15b86ca9a3938574aedfdd8b3586749d4f0a253bce8285",
}
INSTALL = {
    "PiecewiseCATrace.hh": "KinKal/Trajectory/PiecewiseCATrace.hh",
    "CrossingTrace.hh": "KinKal/Fit/CrossingTrace.hh",
    "CrossingCapture.hh": "Offline/Mu2eKinKal/inc/CrossingCapture.hh",
    "CrossingCaptureWriter.hh": "TrkFastSim/inc/CrossingCaptureWriter.hh",
    "CrossingCaptureStatus.hh": "TrkFastSim/inc/CrossingCaptureStatus.hh",
    "CrossingGeometryWriter.hh": "TrkFastSim/inc/CrossingGeometryWriter.hh",
}
PATCH_FILES = ("patch_core.py", "patch_fit.py", "patch_pca.py", "patch_module.py", "patch_geometry.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def patch_modules():
    result = {}
    for filename in PATCH_FILES:
        name = Path(filename).stem
        spec = importlib.util.spec_from_file_location(name, TEMPLATES/filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        result[name] = module
    for name in ("patch_core", "patch_fit", "patch_pca", "patch_geometry", "patch_module"):
        result[name].__spec__.loader.exec_module(result[name])
    return result


def patched(sources):
    modules = patch_modules()
    result = dict(sources)
    result["KinKal/Fit/Track.hh"] = modules["patch_core"].track(sources["KinKal/Fit/Track.hh"])
    result["Offline/Mu2eKinKal/inc/KKStrawXing.hh"] = modules["patch_core"].xing(sources["Offline/Mu2eKinKal/inc/KKStrawXing.hh"])
    for name, module in (("KinKal/Trajectory/PiecewiseClosestApproach.hh", "patch_pca"),
                         ("Offline/Mu2eKinKal/inc/KKFit.hh", "patch_fit"),
                         ("TrkFastSim/src/KinematicLineFitTrace_module.cc", "patch_module")):
        result[name] = modules[module].patched(sources[name])
    return result


def preparation_inputs(base, area):
    if base != BASE or area != AREA or base.resolve() == area.resolve():
        raise ValueError("only the named validated operand base and isolated crossing destination are supported")
    if area.exists() or area.is_symlink():
        raise ValueError("refusing existing crossing capture area")
    sources = {}
    for name, expected in FROZEN.items():
        if digest(base/name) != expected:
            raise ValueError("validated base hash differs: "+name)
        sources[name] = (base/name).read_text()
    for repo, pin in PINS.items():
        if git(base/repo, "rev-parse", "HEAD") != pin:
            raise ValueError("validated base pin differs: "+repo)
        status = git(base/repo, "status", "--porcelain", "--untracked-files=all")
        expected = "M Fit/Track.hh\n?? Fit/IterationTrace.hh" if repo == "KinKal" else ""
        if status != expected:
            raise ValueError("unexpected validated base worktree changes: "+repo+"\n"+status)
    templates = {str(TEMPLATES/name): digest(TEMPLATES/name) for name in (*INSTALL, *PATCH_FILES)}
    files = sorted((base/"TrkFastSim").rglob("*"))
    if any(p.is_symlink() for p in files):
        raise ValueError("TrkFastSim symlink requires explicit isolation review")
    package = {str(p.relative_to(base)): digest(p) for p in files if p.is_file()}
    return sources, templates, package


def prepare(base, area, report_path, validate_only=False):
    report_resolved = report_path.resolve()
    protected = (base.resolve(), area.resolve(), TEMPLATES.resolve(), Path(__file__).resolve().parents[1])
    if any(report_resolved.is_relative_to(root) for root in protected):
        raise ValueError("preparation report must be outside source, template and destination trees")
    with report_path.open("x", encoding="utf-8") as stream:
        report = dict(completed=False, validated_only=bool(validate_only), base=str(base), area=str(area),
                      build_run=False, physics_run=False, runtime_passivity_validated=False,
                      preparer_sha256=digest(__file__), pins=PINS, frozen_sources=FROZEN)
        watched = {}
        try:
            sources, templates, package = preparation_inputs(base, area)
            watched = {str(base/name): sha for name, sha in FROZEN.items()}
            watched.update({str(base/name): sha for name, sha in package.items()})
            watched.update(templates)
            watched[str(Path(__file__).resolve())] = report["preparer_sha256"]
            changes = patched(sources)
            report.update(templates=templates, package_sources=package,
                          changed_paths=[name for name in changes if changes[name] != sources[name]])
            if not validate_only:
                area.mkdir()
                for repo, pin in PINS.items():
                    git(base/repo, "worktree", "add", "--detach", str(area/repo), pin)
                shutil.copytree(base/"TrkFastSim", area/"TrkFastSim")
                for name, text in changes.items():
                    (area/name).write_text(text)
                for name, target in INSTALL.items():
                    shutil.copyfile(TEMPLATES/name, area/target)
                names = sorted(set(FROZEN) | set(package) | set(INSTALL.values()))
                output = {name: digest(area/name) for name in names}
                report["prepared_sources"] = output
                (area/"source-pins.txt").write_text("".join(f"{repo} {pin}\n" for repo, pin in PINS.items()))
                (area/"capture-source-manifest.sha256").write_text("".join(f"{sha}  {name}\n" for name, sha in output.items()))
                for repo in PINS:
                    git(area/repo, "diff", "--check")
            report["completed"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            end = {}
            for path in watched:
                try:
                    end[path] = digest(path)
                except (OSError, ValueError):
                    end[path] = None
            report.update(before_sha256=watched, end_sha256=end, hashes_unchanged=bool(watched) and watched == end)
            if not report["hashes_unchanged"]:
                report["completed"] = False
            stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("completed", "hashes_unchanged", "validated_only", "error", "changed_paths") if k in report}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--area", type=Path, default=AREA)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = prepare(args.base, args.area, args.report, args.validate_only)
    raise SystemExit(0 if result["completed"] else 1)
