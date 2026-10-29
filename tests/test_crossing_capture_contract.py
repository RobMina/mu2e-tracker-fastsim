"""Static instrumentation wiring checks; these are not C++ compilation or native passivity tests."""
from pathlib import Path
import importlib.util
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/"art/crossing_capture"


def text(name):
    return (ART/name).read_text()


@pytest.mark.parametrize("name", ["CrossingCaptureWriter.hh", "CrossingGeometryWriter.hh", "CrossingCaptureStatus.hh"])
def test_writers_check_ready_before_output(name):
    body = text(name)
    assert "bool ready_" in body
    assert "ready_=true" in body or "ready_ = true" in body
    assert "!ready_" in body or "ready_ &&" in body
    assert "Branch failed" in body and "Fill failed" in body


def test_sink_has_no_global_observer_and_failure_does_not_throw():
    body = text("CrossingTrace.hh")
    assert "void fail(char const* message) noexcept" in body
    assert "template<class COPY> void copy(COPY const& callback) noexcept" in body
    assert "catch(...)" in body
    assert "thread_local" not in body
    assert "std::map<void const*, size_t> identities" in body
    assert "pendingRadius" in body and "pendingSpeed" in body


def test_capture_adapter_never_recomputes_fit_or_ca():
    body = text("CrossingCapture.hh")
    for forbidden in ("findTCA(", "updateReference(", "updateState(", "solve(", "weight(", "position3(", "speed("):
        assert forbidden not in body
    start = body.index("if(out.usable)")
    assert start < body.index("ca.particlePoca()") < body.index("ca.dDdP()")


def test_runtime_addresses_are_not_serialized():
    body = text("CrossingCaptureWriter.hh")
    assert 'branch(records_, "object", &record_.object)' in body
    assert 'branch(records_, "partner", &record_.partner)' in body
    assert 'branch(records_, "offset", &record_.offset)' in body
    assert 'branch(records_, "effect_time", &record_.effectTime)' in body
    assert "reinterpret_cast" not in body
    assert 'branch(records_, "pointer"' not in body


def test_discovery_skips_unreached_computations_and_explicitly_rejects_vertical_capture():
    body = text("patch_fit.py")
    assert '"Vertical material discovery not instrumented"' in body
    for stage in range(9):
        assert f"crossingDiscovery<KTRAJ>({stage}," in body
    assert "row.search=crossingSearch; row.hasSearch=true; row.state=crossingState(pca)" in body
    assert "row.object=KinKal::Track<KTRAJ>::crossingCaptureSink()->lookup" in body
    assert "static_cast<double>(strawNBuffer_)" in body


def test_candidate_trajectory_hook_is_at_the_native_reference_update_site():
    body = text("patch_core.py")
    assert "captureCrossingTrajectory(*ptraj,4);\\n      for(auto& effptr : effects_) effptr->updateReference" in body
    assert "captureCrossingTrajectory(*fittraj_,3)" in body
    assert "CrossingReferenceStage crossingStage(crossingCaptureSink(),2)" in body


def test_real_module_patch_guards_diagnostic_initialization_without_old_schema_changes():
    spec = importlib.util.spec_from_file_location("prepare_crossing_contract", ROOT/"scripts/prepare_f1_crossing_capture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mods = module.patch_modules()
    g = mods["patch_geometry"]
    source = (g._INCLUDE+g._MEMBER+g._BEGIN+g._GUARD+g._CALL+
        '          StockTrack::IterationCapture capture(iterationCaptureRequested_ ? &iterations : nullptr, &captureHitOperands<KTRAJ>);\n'+
        g._BUILD+'            iterationWriter_.write(*trk, iterations, iterationCaptureWritten_, iterationCaptureErrors_);\n')
    out = mods["patch_module"].patched(source)
    assert "try { crossingWriter_.beginJob(*tfs); crossingStatus_.beginJob(*tfs); } catch(...) {}" in out
    assert "try { crossingGeometry_.beginJob(*tfs); } catch(...) {}" in out
    assert "iterationCaptureValid_ =" not in out
    assert "iterationCaptureErrors_ +=" not in out
    assert 'crossingStatus_.write(run_,subrun_,event_,iseed,0' in out
    assert 'crossingStatus_.write(run_,subrun_,event_,iseed,1' in out
    assert 'crossingStatus_.write(run_,subrun_,event_,iseed,-1' in out
    assert out.index("StockTrack::IterationCapture capture") < out.index(g._BUILD)
    with pytest.raises(ValueError):
        mods["patch_module"].patched(out)
