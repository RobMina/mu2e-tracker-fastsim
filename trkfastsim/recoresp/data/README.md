# Reco response tables

`ce_loophelix_run1a.npz`: conversion-electron response of stock LoopHelix reconstruction (KKDe fit, OffSpill
digitisation, Run 1A geometry), built with `python3 -m trkfastsim.recoresp.table` from Geant4 CeEndpoint events
(`dts.mu2e.CeEndpoint.MDC2025ba`, sequencers 001430_00000000..00000600, 19,630 events): the Geant4 primary's
StrawGasSteps from its tracker entry, injected into the same events and run through stock digitisation,
reconstruction and EventNtuple. Bins: step rows (`nrow_edges`) x entry tan(dip) (`tandip_edges`); field
`bz` = 1.0117 T at the tracker origin. Validation on independent events is not part of the table.
