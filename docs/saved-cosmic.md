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
rank as degrees of freedom. Bins with fewer than **10 entries in either arm** are omitted from chi-square and
pulls; each arm is normalized within the included bins only. The full histogram and raw-count ratios still retain
those low-count bins. The omitted bin/entry counts and bin mask are saved, and plot titles show omitted entries.
This is a minimum-count safeguard, not a calibrated goodness-of-fit test for sparse paired migrations.
Unsupported shape differences produce an undefined/infinite statistic rather than false agreement. Empty or
zero-variance bins have undefined pulls; fewer than two events give undefined efficiency errors, displayed as NaN.
Nonfinite/out-of-range `p_first` events are counted separately from efficiency bins but retained in total efficiency.
Under/overflow and missing reconstructed momentum are reported outside the displayed histogram range.
The good-track definition is status > 0, not a production CE selection.

The saved 40,991-event single-parent development sample gives 1,948 fast and 1,914 stock good events:
+0.08295 ± 0.05322 percentage points; ratio 1.01776 ± 0.01150. The fast arm uses the **solid-gas density-effect
branch, no radiative losses, and `complete_outer_wall=False`**. The stock arm uses **Geant4 first-passage steps**;
both use stock reconstruction. The labels in both figures specify these historical settings. Neither saved fast
arm is the current-main default (gas-state density effect and radiative losses).

The separate wall-enabled comparison has 1,909 good events: −0.01220 ± 0.05169 percentage points relative to G4;
it must not be labelled as disabled-wall performance. On the common 393-event 2–5 GeV/c cohort, enabled-minus-
disabled relative momentum-proxy width is −0.04307 ± 0.17223 percentage points: no demonstrated narrowing.

This explored parent provides no independent-parent error or held-out confirmation. Reconstructed momentum uses
the earliest in-bounds SID 0–4, not common-surface, truth-matched resolution. The historic saved legacy physics
settings are not a new validation of every current-main default.
