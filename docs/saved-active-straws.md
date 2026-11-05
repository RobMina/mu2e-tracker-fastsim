# Saved CE active-straw recount

This command reads existing products only and never invokes transport, digitisation or reconstruction. Keep event
tables and ntuples outside the public repository.

```bash
python3 -m trkfastsim.validation.active_straws \
  --stock /path/to/ce_g4.root --entry /path/to/ce_entry.root --chain /path/to/ce_chain.root \
  --entry-events /path/to/entry/events.csv.gz --chain-events /path/to/chain/events.csv.gz \
  --numbers /path/to/figures/numbers.json
```

The command selects the same first good track as the saved comparison and counts distinct
`(plane, panel, straw)` with **KinKal state > -2**, including left (-1), null (0) and right (1). It checks hit-row
counts against persisted `nactive`, reproduces the old right-only counts, and leaves the birth/track cohort
unchanged. It writes a new numbers file or extends an existing one. On the saved 2,000-birth sample:

| Paired cohort | Fast straws | Stock straws | Fast − stock (paired error) |
|---|---:|---:|---:|
| Full CE chain, 942 pairs | 38.7091 | 38.7282 | −0.0191 ± 0.2693 |
| Shared-entry control, 985 pairs | 38.4640 | 38.3452 | +0.1188 ± 0.1726 |

The former +0.1730 ± 0.1882 and +0.3218 ± 0.1528 values count only state > 0 and are superseded as all-active
straw claims. They remain explicitly labelled diagnostics in the numerical output. Active rows and distinct
straws coincide on these samples but are not generally interchangeable. These primary-only historical stock
LoopHelix outputs do not establish production OnSpill/pileup parity.
