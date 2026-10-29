"""Source transformation guards only; no compiler, build, or native passivity claim."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/"scripts"))
import prepare_f1_operand_capture as P


@pytest.fixture
def sources():
    return {
        "KinKal/Fit/IterationTrace.hh": "namespace KinKal {\n    std::optional<Weights> weight;\n    double deweight = 1;\n}",
        "KinKal/Fit/Track.hh": """      class IterationCapture {
explicit IterationCapture(std::vector<IterationTrace>* sink) noexcept :
            previousSink_(iterationSink_), previousCapture_(iterationCapture_) {}
          IterationCapture* previousCapture_;
          out.weight = hit->weight();
  template <class KTRAJ> void Track<KTRAJ>::iterate(MetaIterConfig const& miconfig, IterationTrace* trace) {
    updateEffects(miconfig); filter(); setStatus();
}
""",
        "TrkFastSim/inc/IterationCaptureWriter.hh": """#include "KinKal/Fit/IterationTrace.hh"
    TTree *seeds_ = nullptr;
    void beginJob(art::TFileService& fs) {
    }
            fill(effects_);
""",
        "TrkFastSim/src/KinematicLineFitTrace_module.cc": "StockTrack::IterationCapture capture(iterationCaptureRequested_ ? &iterations : nullptr);\nauto trk = build(config_);\n"}


def test_transformation_additive_scoped_and_keeps_physics(sources):
    before = dict(sources)
    changed = P.patched(sources)
    assert sources == before
    track = changed["KinKal/Fit/Track.hh"]
    assert "observer_(sink ? observer : nullptr)" in track
    assert "HitOperandObserver observer_;" in track
    assert "std::optional<HitOperandTrace>&" in track
    assert track.index("out.weight = hit->weight();") < track.index("observer_(*hit,out.hitOperands)")
    assert track.index("varianceScale = miconfig.varianceScale()") < track.index("updateEffects(miconfig)")
    assert "updateEffects(miconfig); filter(); setStatus();" in track
    writer = changed["TrkFastSim/inc/IterationCaptureWriter.hh"]
    assert "Missing straw hit operands" in writer and "Straw hit operand identity differs" in writer
    assert "hitOperands_.write(run_,subrun_,event_,iseed_,n,i,t,e);" in writer
    assert changed["TrkFastSim/src/KinematicLineFitTrace_module.cc"].endswith("auto trk = build(config_);\n")
    assert "std::optional<HitOperandTrace> hitOperands" in changed["KinKal/Fit/IterationTrace.hh"]


@pytest.mark.parametrize("failure", ["missing", "duplicate"])
def test_replace_requires_unique_anchor(failure):
    text = "before" if failure == "missing" else "anchor anchor"
    with pytest.raises(ValueError, match="source anchor is not unique"):
        P.replace(text, "anchor", "replacement")


def test_second_application_is_refused(sources):
    with pytest.raises(ValueError, match="source anchor is not unique"):
        P.patched(P.patched(sources))


def test_header_uses_guarded_cached_reads_only():
    header = (Path(__file__).parents[1]/"art/operand_capture/HitOperandCapture.hh").read_text()
    callback = header.split("template<class KTRAJ> class HitOperandWriter")[0]
    for forbidden in ("updateState(", "updateReference(", ".weight(", "fillDriftInfo(",
                      "unbiasedClosestApproach(", "dXdPar(", "position3(", "findTCA(", "delta()"):
        assert forbidden not in callback
    assert "auto const& ca = sh->closestApproach();" in callback
    assert "auto const& state = ca.tpData();" in callback
    assert "if(out.caUsable){" in callback
    assert "row.active() ? row.dRdP()[j] : std::numeric_limits<double>::quiet_NaN()" in callback
    assert "KinKal::HitOperandTrace out;" in callback
    assert "payload = std::move(out);" in callback
    assert callback.index("if(out.caUsable){") < callback.index("ca.dDdP()[j]")
    assert "noexcept" not in callback


def test_preparation_rejects_paths_and_existing_area_before_git(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "git", lambda *a: pytest.fail("no git operation allowed"))
    with pytest.raises(ValueError, match="named frozen"):
        P.prepare(tmp_path, P.AREA)
    monkeypatch.setattr(P, "BASE", tmp_path/"base")
    monkeypatch.setattr(P, "AREA", tmp_path/"existing")
    P.AREA.mkdir()
    with pytest.raises(ValueError, match="existing"):
        P.prepare(P.BASE, P.AREA)


def test_runtime_driver_uses_original_publisher_include_root():
    text = (Path(P.__file__).parent/"run_f1_operand_pair.sh").read_text()
    assert "PUBLISHER=/exp/mu2e/app/users/ram2aq/analysis/thread-f-publish-20261025/art" in text
    assert 'FHICL_FILE_PATH="$CONFIG:$PUBLISHER:$AREA:${FHICL_FILE_PATH}"' in text
    assert 'test -f "$PUBLISHER/TrkFastSim/fcl/$config.fcl"' in text
    assert 'sha256sum "$PUBLISHER/TrkFastSim/fcl/$config.fcl"' in text
    assert "trace_f1_candidates trace_rejection_control trace_kkline_mc dump_tracks_mc" in text
    assert "trace_kkline_mc_iter_nopd" not in text


def test_no_setup_build_or_run_invocation_in_preparer():
    text = Path(P.__file__).read_text()
    for forbidden in ("muse setup", "muse build", "cmake --build", "scons", "mu2e -c", "build_kinkal_into_area"):
        assert forbidden not in text
