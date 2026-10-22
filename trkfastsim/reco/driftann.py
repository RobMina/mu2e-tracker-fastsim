"""Port of the KinKal drift-state ANNs (Offline Mu2eKinKal DriftANNSHU with the TMVA SOFIE TrainSign / TrainCluster
sessions): dense in -> h -> h -> h -> 1, ReLU, ReLU, ReLU, sigmoid. Weights are read from the Offline .dat files
(`tensor_<name> <size>` followed by the values; SOFIE kernels are [in, out] row-major, out = x @ K + b), from
$MU2E_SEARCH_PATH or the Musing on CVMFS. Inputs as DriftANNSHU::wireHitState:
  sign:    |doca|, cDrift, sigma(doca), TOT drift time, edep * sin(theta)
  cluster: |doca|, cDrift, TOT drift time, edep / (sqrt(max(0.25, 6.25 - rDrift^2)) / sin(theta))
setLR = cluster > cluster_cut and sign > sign_cut (no annprob flag in CHDRIFTEXT); else null.
"""
import os

import jax.numpy as jnp
import numpy as np

DEFAULT_DIRS = ["/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ay/backing/Offline/Mu2eKinKal/data"]
LAYERS = ("dense", "dense1", "dense2", "dense3")
# CHDRIFTEXT DriftANNSHUSettings (sign weights, sign cut, cluster weights, cluster cut) per DriftANN meta-iteration
SCHEDULE = (("TrainSign_Stage0.dat", 0.5, "TrainCluster_Stage0.dat", 0.5),
            ("TrainSign_Stage1.dat", 0.4, "TrainCluster_Stage1.dat", 0.4),
            ("TrainSign_Stage1.dat", 0.3, "TrainCluster_Stage1.dat", 0.3),
            ("TrainSign_Stage1.dat", 0.2, "TrainCluster_Stage1.dat", 0.2),
            ("TrainSign_Stage1.dat", 0.1, "TrainCluster_Stage1.dat", 0.1))


def find(name):
    dirs = [os.path.join(p, "Offline/Mu2eKinKal/data") for p in os.environ.get("MU2E_SEARCH_PATH", "").split(":") if p]
    for d in dirs + DEFAULT_DIRS:
        f = os.path.join(d, name)
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f"driftann: {name} not found in MU2E_SEARCH_PATH or {DEFAULT_DIRS}")


def load(path):
    """SOFIE .dat -> list of (K [in, out], b [out]) in layer order."""
    toks = open(path).read().split()
    t, i = {}, 0
    while i < len(toks):
        name, n = toks[i], int(toks[i + 1])
        t[name.replace("tensor_", "")] = np.array(toks[i + 2:i + 2 + n], np.float32)
        i += 2 + n
    layers = []
    for L in LAYERS:
        b = t[L + "bias0"]
        k = t[L + "kernel0"]
        layers.append((k.reshape(-1, len(b)), b))
    return layers


def mlp(layers, x):
    """x [..., in] float32 -> [...] sigmoid output (float32, as SOFIE)."""
    h = x.astype(jnp.float32)
    for li, (k, b) in enumerate(layers):
        h = h @ jnp.asarray(k) + jnp.asarray(b)
        h = jnp.maximum(h, 0.0) if li < len(layers) - 1 else 1.0 / (1.0 + jnp.exp(-h))
    return h[..., 0]


def inputs(doca, docavar, cdrift, rdrift, totdt, edep, dirdot):
    sint = jnp.sqrt(jnp.maximum(1.0 - dirdot * dirdot, 0.0))
    sp = jnp.stack([jnp.abs(doca), cdrift, jnp.sqrt(jnp.maximum(docavar, 0.0)), totdt, edep * sint], -1)
    plen = jnp.sqrt(jnp.maximum(0.25, 6.25 - rdrift * rdrift)) / jnp.maximum(sint, 1e-6)
    cp = jnp.stack([jnp.abs(doca), cdrift, totdt, edep / plen], -1)
    return sp, cp
