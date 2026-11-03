# Paired per-shard statistic for tracker comparisons

`shardhist` and `shardstats` compare two arms histogram by histogram with one error rule: the variance of the
per-shard difference. They replace the mixed Poisson, event-clustered and legacy band rules for the step, digi,
hit and fit comparisons. Results: [`validation.md`](validation.md); open items: [`limitations.md`](limitations.md).

## Method

A shard is a subrun. Arms made from the same generated primaries share their subruns, and subruns are independent.
`--batches K` splits each subrun into K shards by event number modulo K, for samples with few subruns.

- For bin b, d_sb = n_cand,sb - n_ref,sb, D_b = sum_s d_sb, and var_b = N/(N-1) sum_s (d_sb - mean_b)^2. This one
  variance covers event clustering and the correlation of the arms through their shared primaries. Underflow and
  overflow are bins too. Bins where no shard differs are left out of ndf.
- chi2/ndf is the mean of D_b^2/var_b times (N-3)/(N-1), because D/sigma is a Student t with N-1 degrees of freedom.
- **Null.** Bins of one histogram are correlated, so chi2(ndf) is the wrong reference. The null is 20,000 random sign
  flips of the shard differences. Every histogram uses the same flips.
- **Limit.** Each histogram gets its own limit, with a family-wise rate of 1 % over all histograms of a comparison.
  Gamma fits to each flip distribution put the histograms on a common scale. The 1 % quantile of the per-flip
  minimum p then sets the common p.
- **Size.** A flag means a difference is resolved, not that it is large. Read the size next to every flag:
  - `rms%` is the n-weighted rms relative difference per bin after the noise is subtracted;
  - `total%` is the relative difference of the entries;
  - `value diff %` is the difference of the summed values, with its paired error. For a per-event histogram this is
    the multiplicity itself.
- `--shape` scales the candidate by the ratio of selected events, so that a rate difference does not flag every
  histogram.
- **Unpaired split null** (`--splits ARM`). Half of one arm's shards are compared with the other half. The first-half
  vs second-half and even vs odd splits are the tests; random halves give the reference.
- `--detail KEY` prints the per-bin ratio with its paired error and the pull.

The statistic is conditional on the fixed secondary library. Its finite size shows up as a real difference, not as
noise.

## Keys

| prefix | content |
|---|---|
| `sel/events` | events passing the PrimaryFilter tracker term for any primary muon (the rate) |
| `ev/` | per selected event: primary-muon steps, other steps, eIon, occupied straws, plane extent (last - first plane) of the primary-muon steps |
| `mu/` | per primary-muon step: eIon, path, \|p\|, time, plane, panel, straw |
| `sec/` | per other step: eIon, path, species (e-, e+, non-primary muon, other), place (in a muon straw, same plane, 1-2 planes, farther, no muon), plane distance, place by the event's muon plane extent |
| `straw/` | per occupied straw: summed eIon (all straws, straws with a muon step, secondary-only, muon-only), steps per straw, straw occupancy; in muon-only straws the muon path length and eIon per mm |
| `digi/` | the `digicmp` histograms: panel and straw occupancy, TDC (absolute and relative), TOT, PMP, ADC samples, dt, digis per event |
| `hit/` | fast hit reco on these digis: wire distance, energy, straw hits and combo hits per event; `ARM@stockreco` holds the stored stock hits under the same keys |
| `fit/` | F2 per shard (`--f2`): fits per event, active hits, chi2/ndf; `ARM@kkline` holds the stock KKLine fits (status > 0) of the same events |

## Commands

```bash
# per-shard histograms, one npz per arm
python3 -m trkfastsim shardhist "FAST/fasttrkCL.*.root" --arm fast --outdir NPZ
python3 -m trkfastsim shardhist "G4/sgsteps.*.root" --arm g4 --outdir NPZ
python3 -m trkfastsim shardhist "STOCK/strawhits.*.root" --arm stock --kind digis --select-steps "FAST/fasttrkCL.*.root" \
  --hit-cond hitcond.json --drift-source STOCK_WITH_CONDITIONS.root --stock-hits --f2 --kkline "STOCK/nts.*.root" --outdir NPZ
python3 -m trkfastsim shardhist "FASTDIGI/fastdigi.*.root" --arm fastdigi --kind digis --select-steps "FAST/fasttrkCL.*.root" \
  --hit-cond hitcond.json --drift-source STOCK_WITH_CONDITIONS.root --f2 --outdir NPZ
# comparisons, nulls, per-bin detail
python3 -m trkfastsim shardstats NPZ/*.npz --pairs g4:fast fast_seed2:fast --splits g4 --shape \
  --detail straw/eion_keV sec/place --outdir STATS
```

Each step file is read once. Memory peaks at about 4 GB for the digi kind with `--f2` on 20 groups of 2,700 events.

## Legacy per-straw band

The T-V1b "eIon per straw, binned ratio" used `stats.ratio_band`: the largest |fast/G4 - 1| over 20 bins of the
reference's 5-95 % range, with Poisson errors per straw entry and 2 sigma subtracted. Showers put many straws into
few events, so those errors are too small, and the maximum over 20 bins is not calibrated. On the sample it was
quoted on (`g4ref_v1`, 10 subruns, about 2,800 selected events per arm):

| comparison | band (raw) | band minus 2 sigma |
|---|---:|---:|
| Geant4 subruns 1-10 vs 11-20 | 19.6 % | 5.6 % |
| Geant4 subruns 11-15 vs 16-20 | 16.7 % | 1.8 % |
| fast seed 1 vs fast seed 2, subruns 11-20 | 7.9 % | 1.0 % |
| fast seed 1 vs Geant4, subruns 11-20 | 21.4 % | 7.2 % |
| fast seed 2 vs Geant4, subruns 11-20 | 15.6 % | 2.0 % |
| fast seed 1 vs Geant4, subruns 1-20 | 12.8 % | 0.1 % |

Geant4 against itself fails the 5 % threshold. Two fast seeds against the same Geant4 events differ by 5 points. The
12.4 % recorded on 2026-10-18 is within the metric's own spread; it does not identify a per-straw discrepancy.
