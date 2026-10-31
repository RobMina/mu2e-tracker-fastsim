"""Prepare an isolated p109 passive hit-operand capture area; never build or run physics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


from mu2efastsim import site

# The frozen capture areas are named directories under TRKFS_MUSE_AREAS (site configuration); the placeholder
# only keeps the module importable for the source-transformation tests, and the command line requires the variable.
ROOT = Path(site.setting("TRKFS_MUSE_AREAS") or "/path/to/TRKFS_MUSE_AREAS")
BASE = ROOT/"fastreco_capture_p109_20261026"
AREA = ROOT/"fastreco_operands_p109_20261028"
PINS = {"Offline": "9b3d463371f34731f8515d1a63914ccf070536a8",
        "KinKal": "bb23c26348fc6e5fcbeefb5c7138919ef21f7abc",
        "Production": "4479f3fad37a85ecd9485394198658237961dcc3",
        "mu2e-trig-config": "d3ec315c7e8a626e8cf9c417ab6522a8e9a8f669",
        "EventNtuple": "76ecb8af80c60e98d80a62a29f0082f36c20add5"}
FROZEN = {"KinKal/Fit/Track.hh": "1ca30b9599868ac8e0e915dc8c427484783b2d4fbad771cb463952ad71eecc3b",
          "KinKal/Fit/IterationTrace.hh": "652dca3770281d1287d54c6d0e8eafe8afb98b6422e03b0ce102bcb80baee2cf",
          "TrkFastSim/src/KinematicLineFitTrace_module.cc": "7ba95ad471e9171885e73c30ae71fd8c7da258694f66b5fb51d7567ade00b83d",
          "TrkFastSim/inc/IterationCaptureWriter.hh": "173a6a922fc9db81ffcb359f90ff635dcee527951cd1bca917649523dcbe9cd0"}
PAYLOAD = """  struct HitOperandTrace {
    long long shindex = -1, strawid = -1;
    int caStatus = 6, hitState = -3;
    bool caUsable = false, constrainTOT = false, constrainLong = false, constrainAbsDrift = false;
    double precision = std::numeric_limits<double>::quiet_NaN();
    double doca = std::numeric_limits<double>::quiet_NaN(), docaVariance = std::numeric_limits<double>::quiet_NaN();
    double tocaVariance = std::numeric_limits<double>::quiet_NaN(), lsign = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> particlePoca, sensorPoca, particleDirection, sensorDirection, dDdP, dTdP;
    std::vector<int> rowActive;
    std::vector<double> rowValue, rowMvar, rowPvar, rowDerivative;
  };

"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError("source anchor is not unique: " + before[:100])
    return text.replace(before, after, 1)


def patched(sources):
    """Pure exact-anchor transformation of the four frozen capture files."""
    result = dict(sources)
    name = "KinKal/Fit/IterationTrace.hh"
    text = replace(result[name], "namespace KinKal {\n", "namespace KinKal {\n"+PAYLOAD)
    text = replace(text, "    std::optional<Weights> weight;", "    std::optional<Weights> weight;\n    std::optional<HitOperandTrace> hitOperands;")
    text = replace(text, "    double deweight =", "    double varianceScale = std::numeric_limits<double>::quiet_NaN();\n    double deweight =")
    result[name] = text
    name = "KinKal/Fit/Track.hh"
    text = replace(result[name], "      class IterationCapture {", "      using HitOperandObserver = void (*)(HIT const&, std::optional<HitOperandTrace>&);\n      class IterationCapture {")
    text = replace(text, "explicit IterationCapture(std::vector<IterationTrace>* sink) noexcept :\n            previousSink_(iterationSink_), previousCapture_(iterationCapture_)",
        "explicit IterationCapture(std::vector<IterationTrace>* sink, HitOperandObserver observer=nullptr) noexcept :\n            previousSink_(iterationSink_), previousCapture_(iterationCapture_), observer_(sink ? observer : nullptr)")
    text = replace(text, "          IterationCapture* previousCapture_;", "          IterationCapture* previousCapture_;\n          HitOperandObserver observer_;")
    text = replace(text, "          out.weight = hit->weight();", "          out.weight = hit->weight();\n          if(iterationCapture_ && iterationCapture_->observer_)iterationCapture_->observer_(*hit,out.hitOperands);")
    text = replace(text, "  template <class KTRAJ> void Track<KTRAJ>::iterate(MetaIterConfig const& miconfig, IterationTrace* trace) {",
        "  template <class KTRAJ> void Track<KTRAJ>::iterate(MetaIterConfig const& miconfig, IterationTrace* trace) {\n    captureCopy(trace,[&]{ trace->varianceScale = miconfig.varianceScale(); });")
    result[name] = text
    name = "TrkFastSim/inc/IterationCaptureWriter.hh"
    text = replace(result[name], '#include "KinKal/Fit/IterationTrace.hh"', '#include "KinKal/Fit/IterationTrace.hh"\n#include "TrkFastSim/inc/HitOperandCapture.hh"')
    text = replace(text, "    TTree *seeds_ =", "    HitOperandWriter<KTRAJ> hitOperands_;\n    TTree *seeds_ =")
    text = replace(text, "    void beginJob(art::TFileService& fs) {", "    void beginJob(art::TFileService& fs) {\n      hitOperands_.beginJob(fs);")
    text = replace(text, "            fill(effects_);", "            if(effect_.identity_kind == 1 && !e.hitOperands)throw std::runtime_error(\"Missing straw hit operands\");\n            if(e.hitOperands && (effect_.identity_kind != 1 || e.hitOperands->shindex != effect_.shindex || e.hitOperands->strawid != effect_.strawid))\n              throw std::runtime_error(\"Straw hit operand identity differs\");\n            hitOperands_.write(run_,subrun_,event_,iseed_,n,i,t,e);\n            fill(effects_);")
    result[name] = text
    name = "TrkFastSim/src/KinematicLineFitTrace_module.cc"
    result[name] = replace(result[name], "StockTrack::IterationCapture capture(iterationCaptureRequested_ ? &iterations : nullptr);",
        "StockTrack::IterationCapture capture(iterationCaptureRequested_ ? &iterations : nullptr, &captureHitOperands<KTRAJ>);")
    return result


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def prepare(base, area):
    if base != BASE or area != AREA:
        raise ValueError("only the named frozen base and new isolated destination are supported")
    if area.exists():
        raise ValueError("refusing existing operand capture area")
    sources = {}
    for name, sha in FROZEN.items():
        if digest(base/name) != sha:
            raise ValueError("frozen source hash differs: "+name)
        sources[name] = (base/name).read_text()
    changes = patched(sources)
    for repo, pin in PINS.items():
        if git(base/repo, "rev-parse", "HEAD") != pin:
            raise ValueError("frozen source pin differs: "+repo)
        status = git(base/repo, "status", "--porcelain", "--untracked-files=all")
        expected = "M Fit/Track.hh\n?? Fit/IterationTrace.hh" if repo == "KinKal" else ""
        if status != expected:
            raise ValueError("unexpected dirty frozen source: "+repo+"\n"+status)
    header = Path(__file__).resolve().parents[1]/"art/operand_capture/HitOperandCapture.hh"
    header_hash = digest(header)
    area.mkdir()
    for repo, pin in PINS.items():
        git(base/repo, "worktree", "add", "--detach", str(area/repo), pin)
    shutil.copytree(base/"TrkFastSim", area/"TrkFastSim")
    for name, text in changes.items():
        (area/name).write_text(text)
    shutil.copyfile(header, area/"TrkFastSim/inc/HitOperandCapture.hh")
    (area/"source-pins.txt").write_text("".join(f"{repo} {pin}\n" for repo, pin in PINS.items()))
    names = [*FROZEN, "TrkFastSim/inc/HitOperandCapture.hh"]
    (area/"capture-source-manifest.sha256").write_text("".join(f"{digest(area/name)}  {name}\n" for name in names))
    metadata = {"base": str(base), "area": str(area), "frozen_sources": FROZEN, "pins": PINS,
                "preparer_sha256": digest(__file__), "header_sha256": header_hash,
                "build_run": False, "physics_run": False, "runtime_passivity_validated": False}
    (area/"operand-preparation.json").write_text(json.dumps(metadata, indent=2)+"\n")
    if any(digest(base/name) != sha for name, sha in FROZEN.items()) or digest(header) != header_hash:
        raise ValueError("source changed during preparation; preserve area for inspection")
    print(json.dumps(metadata, indent=2))
    print("SOURCE_PREPARATION_DONE; user build and matching local KinKal installation required.")


if __name__ == "__main__":
    site.require("TRKFS_MUSE_AREAS")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--area", type=Path, default=AREA)
    args = p.parse_args()
    prepare(args.base, args.area)
