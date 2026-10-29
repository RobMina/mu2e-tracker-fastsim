"""Pure, fail-closed source patch for the frozen native piecewise-CA constructor.

Only ``patched(text)`` is public; this module never reads or writes source files.
Exact unique anchors reject source drift, ambiguous matches and repeat patching.
The native loop, including its pointer update and post-increment cap, is retained.
Trace indices distinguish the current pointer from the last solved piece; recorded
statuses precede any cap-induced demotion. A failed trace has only a valid prefix
of solve rows, but its scalar indices and exhaustion flag continue to be updated.
"""

_INCLUDE = '#include "KinKal/Trajectory/ParticleTrajectory.hh"\n'
_TRACE_INCLUDE = '#include "KinKal/Trajectory/PiecewiseCATrace.hh"\n'
_DECLARATION = (
    "      PiecewiseClosestApproach(PTRAJ const& ptraj, STRAJPTR strajptr, "
    "CAHint const& hint, double precision,\n"
    "          unsigned maxiter=10);"
)
_DEFINITION = (
    "  template<class KTRAJ, class STRAJ> "
    "PiecewiseClosestApproach<KTRAJ,STRAJ>::PiecewiseClosestApproach("
    "ParticleTrajectory<KTRAJ> const& ptraj, STRAJPTR strajptr, "
    "CAHint const& hint, double prec,\n"
    "      unsigned maxiter) : "
    "KTCA(ptraj.nearestTraj(hint.particleToca_),strajptr,prec) {"
)
_BODY = """    auto pindex = ptraj.nearestIndex(hint.particleToca_);
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
"""
_TRACED_BODY = """    auto pindex = ptraj.nearestIndex(hint.particleToca_);
    if(trace) trace->initialize(hint.particleToca_, hint.sensorToca_, prec, maxiter, pindex);
    this->findTCA(hint);
    if(trace) trace->solved(pindex, hint.particleToca_, hint.sensorToca_, *this);
    if(maxiter == 0) return;
    // iterate until the local TCA is on the same piece, moving at most maxiter times
    unsigned niter=0;
    while(true){
      auto oldindex = pindex;
      pindex = ptraj.nearestIndex(this->particleToca());
      this->ktrajptr_ = ptraj.indexTraj(pindex);
      if(trace) trace->finalIndex = pindex;
      if(pindex == oldindex || !this->usable()) break;
      if(niter++ >= maxiter){
        if(trace) trace->exhausted = true;
        // overwrite the status if we didn't converge on the piece
        if(this->status() == ClosestApproachData::converged) this->tpdata_.status_ = ClosestApproachData::unconverged;
        break;
      }
      auto nextHint = this->hint();
      this->findTCA(nextHint);
      if(trace) trace->solved(pindex, nextHint.particleToca_, nextHint.sensorToca_, *this);
    }
"""
_REPLACEMENTS = (
    ("include", _INCLUDE, _INCLUDE + _TRACE_INCLUDE),
    (
        "constructor declaration",
        _DECLARATION,
        _DECLARATION.replace(
            "unsigned maxiter=10);", "unsigned maxiter=10, PiecewiseCATrace* trace=nullptr);"
        ),
    ),
    (
        "constructor definition",
        _DEFINITION,
        _DEFINITION.replace(
            "unsigned maxiter)", "unsigned maxiter, PiecewiseCATrace* trace)"
        ),
    ),
    ("native constructor body", _BODY, _TRACED_BODY),
)


def patched(text: str) -> str:
    """Return instrumented source, or raise ValueError without accepting drift."""
    if "PiecewiseCATrace" in text:
        raise ValueError("PiecewiseCATrace already present; refusing repeat or partial patch")
    for label, before, _ in _REPLACEMENTS:
        count = text.count(before)
        if count != 1:
            raise ValueError(f"{label}: expected exactly one exact anchor, found {count}")
    for _, before, after in _REPLACEMENTS:
        text = text.replace(before, after, 1)
    return text
