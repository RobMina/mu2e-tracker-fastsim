"""trkfastsim -- Geant4-free tracker simulation and fast reconstruction for the Mu2e extracted (KPP) configuration.

Stage 2 of the fast-simulation effort, beside crvfastsim; physics primitives come from mu2efastsim
(mu2e-fastsim-core). Planned modules (docs/plan.md):

    geometry   KPP straws from an Offline Tracker geometry dump, with sim and reco alignment applied independently
    steps      straight-line straw-gas crossings -> StrawGasStep table (eIon from mu2efastsim.eloss, G4 delta library)
    filter     the production PrimaryFilter OR on fast StrawGasStep/CrvStep tables
    truth      SimParticle, MCTrajectory and SurfaceStep tables for EventNtuple truth
    reco/      fast digitisation (waveform port and direct hit model), hit reco, time clustering, seed, line fits F1-F5
"""
__version__ = "0.0.1"
