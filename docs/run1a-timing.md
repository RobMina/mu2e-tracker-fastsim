# Run 1A fitted timing and drift residuals

This opt-in step follows the saved measured-wire initializer and the independently validated
[stock drift-response port](run1a-drift-contract.md). It is not a CentralHelix/KinKal replacement,
tracking-efficiency result, momentum-resolution measurement, or calibrated direction classifier.
No truth or stock-track membership is used. The original association and wire fits are not rerun.

## Geometry and time conventions

`helixtiming.prepare_timing` takes the five local helix parameters, fixed z reference, saved local
POCA z, explicit cal/HV endpoints, early end/time, energy-dependent signal speed and frozen mask.
Every selected hit must pass input, finite-interior-wire, analytic local-POCA stationarity and
positive-Hessian checks. Undefined angle/sign, nonidentifiable curvature and failed seeds remain
failed lanes; no hit is removed to rescue a row. Masked fields are zero/false, not observations.

The signal direction is **late to early**, toward the readout endpoint. `hitLine` passes early then
late endpoints, but KinKal constructs the directed segment in reverse argument order. With early
endpoint E, sensor POCA Xs, s=unit(E-L), and calibrated early-end time te:

```
sensor_time = te - dot(E-Xs,s)/signal_speed
signal_speed = 2*halfPropV(straw,1000*energyDep_MeV)
```

This corrects the earlier prose statement that the signal direction was early to late. No existing
physical timing code depended on that statement. The new analytic tests check both early ends.
The driver uses `s_et0/s_et1` selected by `s_eend`, never TOT-corrected `s_time`, stored `s_ptime`,
TOT drift estimates or a second PBT subtraction. `ComboHit::timeVar` is the TOT-drift error variance,
**not independent early-end electronics noise**; it is not added to the drift-distance variance.

For measured local parameters `(x0,y0,tx,ty,omega)` and a unit-charge muon mass hypothesis:

```
g = sqrt(1+tx^2+ty^2)
p = 0.299792458*abs(Bz)*g/abs(omega)    # MeV/c, mm, T
beta = p/hypot(p,105.6583755)
tof(d) = d*(z-z_ref)*g/(beta*299.792458), d in [-1,+1]
drift_dt = sensor_time - t0 - tof(d)
```

The spatial helix/nominal trajectory field remains the initializer's **constant axial Bz=1.0117 T**.
That explicit approximation is separate from the exact detector-origin field used to construct the
global stock drift model. No field-map propagation, energy loss or material scattering is included.
Curvature supplies momentum for beta; no truth momentum or beta=1 fallback is used.

The Lorentz angle uses `q=Xp-Xs`, `r=s cross b`, reverses r if `q dot r<0`, computes
`atan2(q dot r,q dot b)` and folds above pi/2. **r is not normalized.** Signed DOCA is
`sign((s cross pdir) dot (-q))*|q|`. Reversing flight direction reverses TOF and signed DOCA, not
spatial POCA or Lorentz angle. DOCA <=1e-9 mm has undefined angle/sign. The supplied-POCA parallel
guard uses `sin(theta)<=1e-5`; it intentionally does not reproduce KinKal's different iterative
CA denominator guard `sin(theta)^2<1e-5`. These are numerical policies, not detector cuts.

## Fixed-geometry t0 profiles

`helixtiming.profile_t0` evaluates **both** flight directions separately on the same full hit set.
The measured anchor is `median(sensor_time-tof(d))`. The closed offset search is -200..50 ns in
0.5 ns steps, then two 21-point local refinements about the best previous point, giving a final
0.005 ns spacing. All coarse points are evaluated; previous minima are retained. A finite grid
plus local refinement is not a proof of a global minimum or numerical optimizer convergence,
especially at calibration-bin switches. Search-boundary minima retain diagnostics but are not
successful profiles. Negative drift times/radii and extrapolated values are never clipped.

The conditional score is

```
0.5 * sum(((absDOCA - calibrated_drift_radius)/signed_drift_error)^2)
```

This is equivalent in squared value to the stock signed-distance residual when the LR sign is
chosen from the fixed geometry. It uses **signed**, not unsigned, error; negative calibrated radius
is not absolutized. It is not a fitted-track chi-square or a probability. Neither score ordering
nor its difference assigns a direction. No uncertainty/covariance is inferred from the grid spacing.

## Saved development check

`validation.run1a_timing` reads the earlier complete hit/association/wire-fit census, replays fitted
POCA diagnostics, and checks recorded wire-fit input hashes. Explicit endpoints and propagation
probes come from the saved hit-reference conditions tree; it reads no stock track products.
The original reference and oracle's conditions/geometry service configurations agree. Resolved
energy-dependent propagation reproduces the saved stock probes. The drift evaluator is rechecked
against all393276 oracle values before profiling. All input files and implementation sources are hashed.

This is not complete original provenance: the association lacks original input hashes and the
wire-fit report does not bind its output NPZ. The driver checks identities and geometric replay,
and records hashes for this invocation rather than claiming either missing binding.

First1000 events: 13492 hits,235 empty events,773 groups,344 saved seeds/5427 selected hits.
The255 converged wire fits (4395 selected hits) pass timing geometry and both profiles. The89 failed
wire fits and429 empty group lanes remain explicit. Neither direction has a boundary minimum or
best-fit delta-t outside the stock-oracle validation interval -50..200 ns. That interval is
coverage information, not a reconstruction cut.

| Fixed geometry | Negative-z hypothesis | Positive-z hypothesis |
|---|---:|---:|
| Profiled groups | 255 | 255 |
| Median conditional score | 108.39 | 111.40 |
| Median absolute drift pull | 2.381 | 2.362 |
| p90 absolute drift pull | 6.327 | 6.359 |
| Maximum absolute drift pull | 16.293 | 16.095 |

The broad pulls show why fitting t0 alone is insufficient; no residual recut hides this. Both
hypotheses are retained rather than quoting a sample-wide preferred direction. The complete NPZ
payload is array-exact across group batches32/7 and query-cell limits262144/32768. Initial laptop
profile processing was0.322 s, load/oracle check0.175 s; output writing is separate. This is not
stock reconstruction throughput or a GPU measurement. No heldout tuning or new physics generation.

```bash
python3 -m trkfastsim.validation.run1a_timing --hits HITS/numpy.npz \
  --conditions HITS/conditions.json --association ASSOCIATION/diagnostics.npz \
  --wirefit WIREFIT --reference REFERENCE/hits.root --oracle ORACLE/drift-oracle.root \
  --config ORACLE/expanded.fcl --drift-check DRIFT_CHECK/report.json --outdir NEW_TIMING
```

Keep `REFERENCE/expanded.fcl`, oracle sidecars and the completed drift audit with these inputs.
The API and driver include synthetic endpoint/timing, tilted-wire, failure, full-census, provenance,
no-overwrite and exact-batching tests. The joint step below is separate and opt-in; the default
continues to generate timing profiles only. Digitiser optimisation stays deferred.

## Joint geometry/time minimization: first development result

`reco.helixdriftfit.fit_drift_helix` varies the five local geometry parameters and t0, separately
for each flight direction. Every objective and all twelve central-difference perturbations refind
local spatial POCA and recompute propagation, Lorentz angle, measured-curvature beta, radius and
signed-error normalization. The initial LR sign is frozen; current signed DOCA may cross the wire
if the resulting geometry is valid. No LR reassignment, hit addition/removal or post-fit recut occurs.

```
drift_residual = initial_LR * calibrated_radius - current_signed_DOCA
objective = 0.5 * sum((drift_residual/signed_error)^2 + longitudinal_pull^2)
```

This is an **engineering conditional objective**, not a calibrated chi-square or covariance.
The actual piecewise stock calibration is retained, not smoothed to make minimization easier.
Central differences of the actual normalized residual are used, not stock `driftVelocity` as a
substitute for all parameter derivatives. The default budget is40 iterations, dimensionless FD
step1e-3 and equilibrated gradient tolerance1e-6. Full rank6, bounded condition, small final
unconstrained gradient and all-hit validity are required. Improving the objective is insufficient.
Failures retain last accepted finite parameters; physical outputs require convergence and the
initializer's numerical curvature-span guard `abs(omega)*max(1,measured_z_span)>1e-8`. This guard
is not a momentum uncertainty estimate. No direction winner or charge is assigned.

On the same development1000/default40 wire-fit input:

| Joint fit, unchanged support | Negative-z | Positive-z |
|---|---:|---:|
| Attempted hypotheses | 255 | 255 |
| Converged | **105** | **90** |
| Damping-stalled | **131** | **151** |
| Iteration-exhausted | **19** | **14** |
| Hits in converged hypotheses | 1699 | 1388 |
| Median initial absolute drift residual, same hits (mm) | 0.368 | 0.386 |
| Median final absolute drift residual, same hits (mm) | **0.184** | **0.182** |
| p90 final absolute drift residual (mm) | 0.594 | 0.578 |
| Median final absolute drift pull | 1.149 | 1.139 |

All510 objectives decreased, but only195 hypotheses converged: **159 groups/events** have at least
one converged direction,36 have both. The other282 stalled and33 exhausted hypotheses remain
failures, in addition to the earlier89 failed wire-fit and429 empty group lanes in each direction.
These are optimizer censuses, **not tracking efficiency**. Successful subsets differ between
columns; each before/after residual comparison uses precisely the same hits within its column.
The final maximum residual can worsen despite a lower summed objective; no hit was pruned.

There are117/85 spatial sign crossings among the negative/positive converged-hit populations,
with LR held fixed. This is not LR-ambiguity resolution: negative calibrated radii are allowed by
stock and kept. No converged fit hit lies outside the oracle dt validation domain. The many
stalls require diagnosis; nonsmooth calibration/angle behavior is a possible mechanism, not an
established explanation for every failure. No thresholds were relaxed to improve this count.

The complete saved arrays are exact between group batches32/17, including failure statuses and
last iterates. Cache `driftfit-1000-20261106-final` retains the final default report and payload;
`driftfit-1000-batch17-20261106` retains the alternate batch. Initial measured laptop processing
was72.05 s; the repeated default run was37.92 s and batch17 was55.80 s, with load/oracle checks and
output writes separate. These are variable development CPU timings, not a speedup or GPU claim.

```bash
# Same arguments as the timing command above, adding the opt-in joint stage:
python3 -m trkfastsim.validation.run1a_timing ... --fit-drift --outdir NEW_DRIFTFIT
```

Final focused validation: **673 passing tests**, warnings treated as errors, including80 timing,
62 joint-fit and43 driver tests. Joint tests include noiseless recovery, an independent noisy
SciPy geometry oracle, fixed LR/membership, numerical failure modes, exact batches and the final
near-straight physical-output guard. Reviews prompted stricter census dtypes, unit-wire geometry
binding and that final curvature guard. No CI, new C++ build or simulation/digitisation run.

## Follow-up: derivative-scale qualification

[Read-only final-state diagnosis](run1a-drift-diagnostic.md) replays all510 attempted hypotheses
without changing any saved status, selected hit or LR.231 of282 stalled rows have offset-regression
coefficient changes in the default finite-difference stencil;279 cross native time cells. These
are overlapping observations, not per-row causal assignments.271 stalls have a sampled decrease
using the smaller-stencil direction, versus25 using the default direction above the same heuristic
arithmetic screen. No failed row was promoted to success.

Crucially, only13 of the195 recorded successes pass the same gradient threshold at the smallest
stencil. The original labels remain faithful to their configured test, but are **not stencil-stable
stationarity certificates**. Most alternative trial decreases are tiny; the smallest stencil is
not assumed to be an exact derivative. The diagnostic also finds pointwise LR improvements but
applies none and assigns no direction. Complete arrays repeat exactly at group batches32/17;
new focused validation is872 passes across16 files, including44 diagnostic/35 driver tests.

## Separate analytic-derivative candidate

[Analytic candidate details](run1a-stabledrift.md): from the identical original seeds/support/LR,
307/510 hypotheses pass analytic stationarity plus two adjacent same-branch FD checks, versus20
saved baseline final states under that **identical** criterion. The earlier195 labels are unchanged
and use a different criterion. Candidate213/255 eligible groups have any passing direction,94both.
134 consistency failures,48 boundary states,17 budget failures and4 stalls remain explicit. All510
objectives decrease from their seeds;411 are below the baseline final values and99 above, mostly
by tiny amounts. This is numerical progress, not tracking efficiency or a resolution measurement.

The baseline fitter, source response and support remain unchanged; no LR or direction is assigned.
Complete candidate/trace/baseline-check arrays repeat exactly at batches32/17;1010 focused tests
pass across19 files. No new C++ build, physics generation, heldout tuning or CI.
Next: remaining consistency/boundary failures, measured LR/direction, multiple tracks/outliers,
material/covariance and saved-stock CentralHelix comparison. Physics performance remains unvalidated.
