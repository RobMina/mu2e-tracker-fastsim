# Saved paired cosmic plots

This command reads an existing paired table only; it never invokes transport or reconstruction. Plotting requires
matplotlib in the analysis environment. Keep event tables outside the public repository.

```bash
python3 -m trkfastsim.validation.cosmic_saved /path/to/events.csv.gz --outdir /path/to/figures
```

The saved `legacy_*` and `g4_*` columns and complete event denominator retain missing/no-good-track outcomes.
Outputs are `cosmic_trk.png` (efficiency, reconstructed momentum, active hits and chi2/ndf, each with fast/G4 ratio),
`cosmic_pulls.png` (normalized-shape pulls), and `numbers.json` (numerical histograms, errors and covariances).
Errors use paired event indicators; shape chi-square uses the normalized-shape covariance pseudoinverse with its
rank as degrees of freedom. Unsupported shape differences produce an undefined/infinite statistic rather than
false agreement. Empty/zero-variance bins have undefined pulls. Under/overflow and missing momentum are reported
outside the displayed range. The good-track definition is status > 0, not a production CE selection.

The saved 40,991-event single-parent development sample gives 1,948 fast and 1,914 stock good events:
+0.08295 ± 0.05322 percentage points; ratio 1.01776 ± 0.01150. The separate wall-enabled comparison is
−0.01220 ± 0.05169 percentage points and must not be labelled as disabled-wall performance.

This explored parent provides no independent-parent error or held-out confirmation. Reconstructed momentum uses
the earliest in-bounds SID 0–4, not common-surface, truth-matched resolution. The historic saved legacy physics
settings are not a new validation of every current-main default.
