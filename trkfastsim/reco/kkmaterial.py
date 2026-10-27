"""Straw material model of KinKal for the F1 fit: KinKal MatEnv effective Z/A of the Mu2e straw-wall and straw-gas
mixtures, DetMaterial::scatterAngleVar (Lynch-Dahl, NIM B58 (1991)) and KKStrawMaterial::pathLengths.

Sources (Offline/KinKal at the off/ workDir): KinKal/MatEnv/{DetMaterial.cc, MtrPropObj.cc}, Offline/TrackerConditions/
data/{MaterialsList,ElementsList}.data, Offline/KinKalGeom/{src/KKStrawMaterial.cc, fcl/prolog.fcl}.
Functions take jax.numpy arrays (jit/vmap safe) and also work on numpy inputs.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ALPHA = 1.0 / 137.036
MMU = 105.6583755

# element (Z, A) from ElementsList.data
_EL = {"H": (1, 1.007940), "C": (6, 12.011000), "O": (8, 15.999400), "Al": (13, 26.981539), "Ar": (18, 39.948000),
       "Au": (79, 196.966540)}


def _by_atoms(atoms):
    """Element mass fractions of a compound given by atom counts (MtrPropObj::AddElement(element, nAtoms))."""
    amol = sum(n * _EL[e][1] for e, n in atoms.items())
    return {e: n * _EL[e][1] / amol for e, n in atoms.items()}


def _mix(parts):
    """Mass-fraction mix of element mass-fraction dicts (AddElement/AddMaterial by fraction)."""
    out = {}
    for frac, comp in parts:
        for e, w in comp.items():
            out[e] = out.get(e, 0.0) + frac * w
    return out


def effective_za(massfrac):
    """MtrPropObj::getZ/getA for a mixture: averages weighted by atoms per volume (n_i ~ w_i / A_i)."""
    n = {e: w / _EL[e][1] for e, w in massfrac.items()}
    tot = sum(n.values())
    return sum(n[e] * _EL[e][0] for e in n) / tot, sum(n[e] * _EL[e][1] for e in n) / tot


MYLAR = _by_atoms({"C": 10, "O": 4, "H": 8})
CO2 = _by_atoms({"C": 1, "O": 2})
# MaterialsList.data: straw-wall 1.4325 g/cm3, 96.95 % Mylar, 1.80 % Au, 1.25 % Al by mass;
# straw-gas 0.0016980 g/cm3, 78.304 % Ar, 21.696 % CO2 by mass
MATERIALS = {
    "straw-wall": {"density": 1.4325, "massfrac": _mix([(0.9695, MYLAR), (0.0180, {"Au": 1.0}), (0.0125, {"Al": 1.0})]),
                   "state": "solid"},
    "straw-gas": {"density": 0.0016980, "massfrac": _mix([(0.78304, {"Ar": 1.0}), (0.21696, CO2)]), "state": "gas"},
}
# KinKalGeom prolog: SolidScatteringFraction, GasScatteringFraction
SCATTERFRAC = {"solid": 0.999999, "gas": 0.9999999}
# DetMaterial::scatterAngleVar caches vfactor and sig2factor in function statics, so the scatter fraction of the first
# material ever evaluated in the job is used for all; in KKLine that is the straw wall (findXings pushes wall first)
STATIC_SCATTERFRAC = SCATTERFRAC["solid"]
# p109 DetMaterial: each material's own fraction (gas 0.9999999); F1_GASFRAC=p109 selects it, default keeps the static one
GAS_SCATTERFRAC = SCATTERFRAC["gas"] if __import__("os").environ.get("F1_GASFRAC", "static") == "p109" else STATIC_SCATTERFRAC

# tracker_v7 straw geometry (mm); StrawProperties inner radius = outer - wall thickness
STRAW = {"orad": 2.5235, "wall": 0.015, "wrad": 0.0125}
STRAW["irad"] = STRAW["orad"] - STRAW["wall"]


def material_constants(name):
    """DetMaterial cached constants: density [g/mm^3], chic2 factor, chia2_1, chia2_2."""
    m = MATERIALS[name]
    z, a = effective_za(m["massfrac"])
    return {"z": z, "a": a, "density": m["density"] / 1000.0, "chic2": 15.7 * z * (z + 1) / a,
            "chia2_1": 2.007e-5 * z ** (2.0 / 3.0), "chia2_2": 3.34 * (z * ALPHA) ** 2}


_MC = {k: material_constants(k) for k in MATERIALS}


def scatter_angle_var(name, mom, pathlen, mass=MMU, scatterfrac=STATIC_SCATTERFRAC):
    """DetMaterial::scatterAngleVar: projected scattering angle variance [rad^2] for path length pathlen [mm]."""
    c = _MC[name]
    beta = mom / jnp.sqrt(mom * mom + mass * mass)
    invmom2, invb2 = 1.0 / mom ** 2, 1.0 / beta ** 2
    path = jnp.abs(pathlen) * c["density"]
    chic2 = c["chic2"] * path * invb2 * invmom2
    chia2 = c["chia2_1"] * (1.0 + c["chia2_2"] * invb2) * invmom2
    v = 0.5 / (1.0 - scatterfrac) * chic2 / chia2
    vs = jnp.where(v > 0, v, 1.0)
    sig2 = 1.0 / (1.0 + scatterfrac ** 2) * chic2 * ((1.0 + vs) * jnp.log1p(vs) / vs - 1.0)
    return jnp.where(v > 0, jnp.maximum(sig2, 0.0), 0.0)


def _segment_area(doca, r):
    th = 2.0 * jnp.arccos(jnp.clip(doca / r, -1.0, 1.0))
    return jnp.where(doca < r, 0.5 * r * r * (th - jnp.sin(th)), 0.0)


def _angle_factor(dirdot):
    return 1.0 / jnp.sqrt(jnp.maximum(1.0 - dirdot * dirdot, 0.01))


def path_lengths(doca, docavar, dirdot, nsig, irad=STRAW["irad"], orad=STRAW["orad"], wall=STRAW["wall"]):
    """KKStrawMaterial::pathLengths -> (wallpath, gaspath) [mm]: segment-area average over |DOCA| +- nsig sigma, or the
    straw average (pi/2 irad gas, pi wall) when 2 nsig sigma > irad; times 1/sin(angle to the straw), capped at 10."""
    dr = nsig * jnp.sqrt(jnp.maximum(docavar, 0.0))
    ad = jnp.abs(doca)
    mind = jnp.maximum(0.0, jnp.minimum(ad, irad) - dr)
    imax = jnp.minimum(irad, ad + dr)
    omax = jnp.minimum(orad, ad + dr)
    agas = _segment_area(mind, irad) - _segment_area(imax, irad)
    gas = jnp.where(agas > 0, agas / jnp.where(imax > mind, imax - mind, 1.0), 0.0)
    awall = _segment_area(mind, orad) - _segment_area(omax, orad) - agas
    wallp = jnp.where(awall > 0, awall / jnp.where(omax > mind, omax - mind, 1.0), 0.0)
    avg = 2.0 * dr > irad
    gas = jnp.where(avg, 0.5 * np.pi * irad, gas)
    wallp = jnp.where(avg, np.pi * wall, wallp)
    af = _angle_factor(dirdot)
    return wallp * af, gas * af


def straw_scatter_var(doca, docavar, dirdot, nsig, mom, mass=MMU):
    """Projected scattering variance of one straw crossing (ElementXing::materialEffects: wall + gas summed)."""
    w, g = path_lengths(doca, docavar, dirdot, nsig)
    return scatter_angle_var("straw-wall", mom, w, mass) + scatter_angle_var("straw-gas", mom, g, mass, GAS_SCATTERFRAC)


# ---------------------------------------------------------------------------------------------------------------------
# Energy loss (DetMaterial::energyLoss / energyLossVar, IonizationEnergyLossMode 1 = Moyal mean): ionisation constants
# of the mixtures as MtrPropObj::ComputeIonisationParam / ComputeOtherParams, element constants as
# ElmPropObj::ComputeIonisationParam (I = 16 Z^0.9 eV, shell-correction polynomials). Muons: no radiation term.
E_MASS = 0.510998910
DGEV = 15.3536                         # DetMaterial::_dgev (MeV per g/mm^2 path, Z/A = 1, beta = 1)
TWOLN10 = 2.0 * np.log(10.0)
BG2LIM = 0.0169
_NA = 6.02214076e23
_HBARC = 197.3269804e-12               # MeV mm
_RE = 2.8179403262e-12                 # mm
_STP_T, _T_MAT = 273.15, 273.15 + 20.0  # MaterialsList temperature column 20.0 (C), pressure 1.0 atm


def ionisation_constants(name):
    m = MATERIALS[name]
    rho = m["density"]                                            # g/cm^3
    ne_i = {e: _NA * rho * 1e-3 * w / _EL[e][1] * _EL[e][0] for e, w in m["massfrac"].items()}   # electrons / mm^3
    na_i = {e: _NA * rho * 1e-3 * w / _EL[e][1] for e, w in m["massfrac"].items()}               # atoms / mm^3
    ne = sum(ne_i.values())
    I_el = {e: 16.0 * _EL[e][0] ** 0.9 * 1e-6 for e in m["massfrac"]}                           # MeV
    I = np.exp(sum(ne_i[e] * np.log(I_el[e]) for e in ne_i) / ne)
    shell = np.zeros(3)
    for e in na_i:
        r = I_el[e] / E_MASS
        r2 = r * r
        shell += na_i[e] * np.array([(1.10289e5 + 5.14781e8 * r) * r2, (7.93805e3 - 2.22565e7 * r) * r2,
                                     (-9.92256e1 + 2.10823e5 * r) * r2])
    shell /= ne
    C = 1.0 + np.log(I * I / (4 * np.pi * _HBARC ** 2 * _RE * ne))
    if m["state"] == "solid":
        icase = 0 if I < 0.1 else 1
        x0 = 0.2 if C < (3.681, 5.215)[icase] else 0.326 * C - (1.0, 1.5)[icase]
        x1, mpow = (2.0, 3.0)[icase], 3.0
    else:
        climi, x0v, x1v = (10., 10.5, 11., 11.5, 12.25, 13.804), (1.6, 1.7, 1.8, 1.9, 2.0, 2.0), (4.0,) * 5 + (5.0,)
        icase = 5
        while icase > 0 and C < climi[icase]:
            icase -= 1
        x0, x1, mpow = x0v[icase], x1v[icase], 3.0
        parcorr = np.log(_STP_T / _T_MAT)                         # log(rho / rho_STP) at 20 C, 1 atm
        C -= parcorr
        x0 -= parcorr / TWOLN10
        x1 -= parcorr / TWOLN10
    a = TWOLN10 * (C / TWOLN10 - x0) / (x1 - x0) ** mpow
    delta0 = 2 * np.pi * E_MASS * _RE ** 2 * ne * 10.0 / rho       # MtrPropObj::_dEdxFactor (used as delta0 by DetMaterial)
    z, aa = effective_za(m["massfrac"])
    return {"I": I, "C": C, "x0": x0, "x1": x1, "m": mpow, "a": a, "delta0": delta0, "shell": shell, "za": z / aa,
            "density": rho / 1000.0}


_IC = {k: ionisation_constants(k) for k in MATERIALS}


def _density_correction(c, bg2):
    x = jnp.log(bg2) / TWOLN10
    low = jnp.where(c["delta0"] > 0, c["delta0"] * 10.0 ** (2 * (x - c["x0"])), 0.0)
    high = TWOLN10 * x - c["C"] + jnp.where(x < c["x1"], c["a"] * jnp.maximum(c["x1"] - x, 0.0) ** c["m"], 0.0)
    return jnp.where(x < c["x0"], low, high)


def _shell_correction(c, bg2):
    return sum(c["shell"][k] / bg2 ** (k + 1) for k in range(3))      # bg2 > BG2LIM for every muon of interest


def energy_loss(name, mom, pathlen, mass=MMU):
    """DetMaterial::energyLoss (Moyal mean, NEGATIVE energy change, MeV) and its variance (Moyal rms^2)."""
    c = _IC[name]
    E = jnp.sqrt(mom * mom + mass * mass)
    beta = mom / E
    gamma = E / mass
    beta2, bg2 = beta * beta, (beta * gamma) ** 2
    xi = DGEV * c["za"] * c["density"] * jnp.abs(pathlen) / beta2
    xis = jnp.where(xi > 0, xi, 1.0)
    dp = (jnp.log(2.0 * E_MASS * bg2 / c["I"]) + jnp.log(xis / c["I"]) - beta2 + 0.200
          - _density_correction(c, bg2) - _shell_correction(c, bg2)) * -xi
    mean = -(jnp.abs(dp) + xi * (0.57721566490153286 + np.log(2.0)))
    var = (np.pi / np.sqrt(2.0) * xi) ** 2
    return jnp.where(xi > 0, mean, 0.0), var


def straw_eloss(wall, gas, mom, mass=MMU):
    """ElementXing::materialEffects for one straw: momentum change dmom = dE E/p (negative) and its variance."""
    e1, v1 = energy_loss("straw-wall", mom, wall, mass)
    e2, v2 = energy_loss("straw-gas", mom, gas, mass)
    dmdE = jnp.sqrt(mom * mom + mass * mass) / mom
    return (e1 + e2) * dmdE, (v1 + v2) * dmdE * dmdE
