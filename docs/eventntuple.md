# EventNtuple output of the fast reco

`reco-ntuple` (`trkfastsim/reco/ntuple.py`, track-table contract in `tracktable.py`) writes `EventNtuple/ntuple` with
the stock branch names, field order, types, nesting and units (mm, ns, MeV/c, detector frame), driven by a stored
stock schema. Every event of the input is written (also those without tracks); by default only good fits, as stock
(`--all-fits` adds failed fits with `goodfit` 0, `status` -1).

- Layout: uproot cannot write ROOT split-class TTree branches or `std::vector<std::vector<T>>`, so the tree is an
  RNTuple. `tree["evtinfo"].array()`, `tree["evtinfo/run"]`, `tree["trk"].array()` (a record of `trk.<leaf>` lists)
  and `tree["trksegs"|"trkhits"|"trksegpars_kl"|"trkmats"].array()` give the same awkward types as on a stock file.
- Split-branch paths: `tree["trk/trk.<leaf>"].array()` (crvextrap `load_quality` and most trk readers) returns the
  stock jagged array because the `trk` record carries one extra trailing field `trk`, the same per-track lists under
  the bare leaf names. `ntuple.without_alias` drops it; `--no-split-alias` writes the bare stock layout.
- Not supported (uproot 5.7 RNTuple reading): `tree.arrays(["trk", "trksegs", ...])`, as pyutils `Importer` calls it,
  returns empty records for every record field (`tree.arrays(filter_name="trk*")` works); `tree.arrays(["trk.nactive"])`
  nests the result under `trk`; `"trk/trk.status" in tree.keys()` is False. ROOT C++ reading of these files (field
  names containing dots) is not checked.
- Not written: the MC branches (`trkmcsim`, `trksegsmc`, `trkhitsmc`, `evtinfomc`), CRV and DS surfaces in `trksegs`
  (stock has the ToCRV extrapolation, sid 201-277), `trk.nseg` (KinKal piecewise segment count, sentinel).
- Unfilled fields hold -9999.0 (floats), -9999 (integers) or False (booleans). The per-field coverage (filled,
  approx, constant, sentinel), the fit type, the stock conditioning, the package commit and a content hash are in
  `EventNtuple/fastreco_provenance` (JSON string).
- `trksegs`: every intersection of the fitted line with TT_Front, TT_Back (r < 950 mm) and TT_Outer; no CRV surfaces.
  The momentum is the fit momentum at every crossing (no energy loss along the line): stock is lower by about
  0.1 MeV/c at the TT_Outer entry.
- `--fit f1` is stock-conditioned: stock LineFinder seed, stock hit list and stock straw-crossing list, single-seed
  single-fit events only. `--fit f2` runs from digis but has a fixed momentum (2000 MeV/c) and no material.
- `scripts/ntuple_validate.sh` runs the three fits and `reco-ntcompare` against the stock EventNtuple of the same digis.
