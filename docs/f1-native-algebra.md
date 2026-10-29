# Native captured-operand numerical audits

These diagnostics validate bounded parts of a passive capture. Neither implements
an independent F1 measurement/material model or certifies fit acceptance.

```bash
python3 scripts/f1_native_decision_audit.py --check CHECK.json --out NEW_DECISION.json
python3 scripts/f1_native_algebra_audit.py --check CHECK.json --out NEW_ALGEBRA.json
```

Both require the complete successful passivity gate, all four evidence hashes,
highest-cycle populations, full integer identity joins and supported payloads. They
refuse an existing output and record explicit exclusions. Numerical mismatch is a
failure, never a request to regularize covariance or relax a fit threshold.

The decision audit recomputes raw candidate-minus-selected-reference steps and
endpoint quadratic forms in native KinematicLine coordinates. It checks threshold
exceedance, not full status precedence: covariance, gap, chi2 and exception behavior
are not independently replayed.

The algebra audit starts from captured prior weights and uses captured ordered
measurement weights and material parameter/covariance changes. Forward-after and
backward-before material caches combine into each downstream candidate piece; the
front candidate combines final backward weights with the original forward prior.
Consecutive materials retain their parameter cache, including across inactive
slots. Weight-to-parameter conversion is demanded only on material lanes without
that cache; covariance-to-weight conversion runs only on material lanes. Inactive,
measurement and padded lanes do not trigger discarded inversions. Iterations are
batched; Python loops only over effect slots in the recursive filter and fixed directions.

Strict component checks use rtol=1e-5 and atol=1e-7. Differences divided by captured
parameter errors are reported separately and never override a failed strict check.
NumPy LU does not reproduce ROOT InvertChol/Bunch-Kaufman floating-point arithmetic.
The preserved optional `--inverse chol` is an eigentest/Cholesky sensitivity path:
nonpositive-eigenvalue lanes use ordinary inverse, but an actual Cholesky failure
following a positive eigentest aborts. It is not a native-exact port.

The separate `--inverse chol-factor` uses actual finite, positive Cholesky pivots
on mirrored lower-triangle symmetric storage. Failed-factor lanes use ordinary
NumPy inverse of the original symmetric operand; successful lanes retain the
factor path. It reports factor-success/fallback counts, including padded identity
operands in the final piece assembly. Fixed six-component loops are vectorized
over all matrices. No eigenvalue clipping, ridge or parameter regularization is
applied. Singular/nonfinite fallback operands fail. This differs from native ROOT
arithmetic and its Bunch-Kaufman fallback, so improvement is sensitivity evidence,
not bit-parity or closure. Both earlier modes and their strict thresholds remain
unchanged.

First differing F1 operands require a separate F1 iteration trace and comparison;
passing a native-input replay cannot establish that the fitter generates those
same operands. Missing iteration targets, unrecorded construction offsets, transit
times and exact detector geometry remain outside these audits.
