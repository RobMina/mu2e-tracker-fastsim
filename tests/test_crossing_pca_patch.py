"""Exact-text patch tests only; no native compilation or surrogate solver."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "art" / "crossing_capture" / "patch_pca.py"
SPEC = importlib.util.spec_from_file_location("crossing_pca_patch", PATCH_PATH)
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)

SOURCE = """#ifndef KinKal_PiecewiseClosestApproach_hh
#define KinKal_PiecewiseClosestApproach_hh
#include "KinKal/Trajectory/ClosestApproach.hh"
#include "KinKal/Trajectory/ParticleTrajectory.hh"
#include <ostream>
namespace KinKal {
  template<class KTRAJ, class STRAJ> class PiecewiseClosestApproach : public ClosestApproach<KTRAJ,STRAJ> {
    public:
      using PTRAJ = ParticleTrajectory<KTRAJ>;
      using KTCA = ClosestApproach<KTRAJ,STRAJ>;
      using STRAJPTR = std::shared_ptr<STRAJ>;
      PiecewiseClosestApproach(PTRAJ const& ptraj, STRAJPTR strajptr, CAHint const& hint, double precision,
          unsigned maxiter=10);
  };
  template<class KTRAJ, class STRAJ> PiecewiseClosestApproach<KTRAJ,STRAJ>::PiecewiseClosestApproach(ParticleTrajectory<KTRAJ> const& ptraj, STRAJPTR strajptr, CAHint const& hint, double prec,
      unsigned maxiter) : KTCA(ptraj.nearestTraj(hint.particleToca_),strajptr,prec) {
    // find the CA on the piece holding the hint time
    auto pindex = ptraj.nearestIndex(hint.particleToca_);
    this->findTCA(hint);
    if(maxiter == 0) return;
    // iterate until the local TCA is on the same piece, moving at most maxiter times
    unsigned niter=0;
    while(true){
      auto oldindex = pindex;
      pindex = ptraj.nearestIndex(this->particleToca());
      this->ktrajptr_ = ptraj.indexTraj(pindex);
      if(pindex == oldindex || !this->usable()) break;
      if(niter++ >= maxiter){
        // overwrite the status if we didn't converge on the piece
        if(this->status() == ClosestApproachData::converged) this->tpdata_.status_ = ClosestApproachData::unconverged;
        break;
      }
      this->findTCA(this->hint());
    }
  }
}
#endif
"""


def test_optional_api_and_native_algorithm_preserved():
    result = PATCH.patched(SOURCE)
    assert result.count('#include "KinKal/Trajectory/PiecewiseCATrace.hh"') == 1
    assert "unsigned maxiter=10, PiecewiseCATrace* trace=nullptr);" in result
    assert "unsigned maxiter, PiecewiseCATrace* trace) : KTCA(" in result
    assert result.count("this->findTCA(") == SOURCE.count("this->findTCA(") == 2
    assert result.count("this->hint()") == SOURCE.count("this->hint()") == 1
    restored = "\n".join(line for line in result.split("\n") if "if(trace)" not in line)
    restored = restored.replace('#include "KinKal/Trajectory/PiecewiseCATrace.hh"\n', "")
    restored = restored.replace(", PiecewiseCATrace* trace=nullptr", "")
    restored = restored.replace(", PiecewiseCATrace* trace", "")
    restored = restored.replace(
        "      auto nextHint = this->hint();\n      this->findTCA(nextHint);",
        "      this->findTCA(this->hint());",
    )
    assert restored == SOURCE


def test_capture_order_and_cap_only_exhaustion():
    result = PATCH.patched(SOURCE)
    ordered = [
        "auto pindex = ptraj.nearestIndex(hint.particleToca_);",
        "if(trace) trace->initialize(hint.particleToca_, hint.sensorToca_, prec, maxiter, pindex);",
        "this->findTCA(hint);",
        "if(trace) trace->solved(pindex, hint.particleToca_, hint.sensorToca_, *this);",
        "if(maxiter == 0) return;",
        "this->ktrajptr_ = ptraj.indexTraj(pindex);",
        "if(trace) trace->finalIndex = pindex;",
        "if(pindex == oldindex || !this->usable()) break;",
        "if(niter++ >= maxiter){",
        "if(trace) trace->exhausted = true;",
        "if(this->status() == ClosestApproachData::converged) this->tpdata_.status_ = ClosestApproachData::unconverged;",
        "auto nextHint = this->hint();",
        "this->findTCA(nextHint);",
        "if(trace) trace->solved(pindex, nextHint.particleToca_, nextHint.sensorToca_, *this);",
    ]
    positions = [result.index(token) for token in ordered]
    assert positions == sorted(positions)
    assert result.count("trace->exhausted") == 1
    assert "trace->solutionIndex =" not in result


@pytest.mark.parametrize("label,before,after", PATCH._REPLACEMENTS)
@pytest.mark.parametrize("copies", [0, 2])
def test_every_anchor_must_be_unique(label, before, after, copies):
    altered = SOURCE.replace(before, before * copies)
    with pytest.raises(ValueError, match="expected exactly one exact anchor"):
        PATCH.patched(altered)


@pytest.mark.parametrize(
    "old,new",
    [
        ("niter++ >= maxiter", "++niter >= maxiter"),
        ("if(maxiter == 0) return;", "if(maxiter==0) return;"),
        ("this->findTCA(this->hint());", "this->findTCA(hint);"),
        ("unsigned maxiter=10", "unsigned maxiter=11"),
        ("\n", "\r\n"),
    ],
)
def test_native_drift_is_rejected(old, new):
    with pytest.raises(ValueError, match="expected exactly one exact anchor"):
        PATCH.patched(SOURCE.replace(old, new))


def test_repeated_or_partial_patch_is_rejected():
    with pytest.raises(ValueError, match="already present"):
        PATCH.patched(PATCH.patched(SOURCE))
    with pytest.raises(ValueError, match="already present"):
        PATCH.patched(SOURCE + '\n#include "KinKal/Trajectory/PiecewiseCATrace.hh"\n')


def test_unrelated_source_is_preserved():
    source = "// leading marker\n" + SOURCE + "// trailing marker\n"
    result = PATCH.patched(source)
    assert result.startswith("// leading marker\n")
    assert result.endswith("// trailing marker\n")


def test_trace_header_has_no_geometry_or_derivative_reads():
    header = PATCH_PATH.with_name("PiecewiseCATrace.hh").read_text()
    for call in ("ca.status()", "ca.usable()", "ca.particleToca()", "ca.sensorToca()"):
        assert header.count(call) == 1
    assert header.count("ca.") == 4
    assert "isusable ? ca.particleToca() : std::numeric_limits<double>::quiet_NaN()" in header
    assert "isusable ? ca.sensorToca() : std::numeric_limits<double>::quiet_NaN()" in header
    assert "catch(...) {\n        failed = true;" in header
    assert header.count("noexcept") == 2
    assert header.index("solutionIndex = index;") < header.index("if(failed) return;")
    assert header.index("usable.reserve(count);") < header.index("solutionIndices.push_back(index);")
