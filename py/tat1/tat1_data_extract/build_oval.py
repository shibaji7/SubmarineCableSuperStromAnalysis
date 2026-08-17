"""
build_oval.py
--------------
Auroral oval (Starkov, 1994a,b, as consolidated by Sigernes et al. 2011,
J. Space Weather Space Clim.) driven by digitized AL index values from the
Feb 11, 1958 storm, for the TAT-1 manuscript.

Method note (important, see figure caption / notes-for-user):
  True Holzworth & Meng (1975) oval-shape representation requires the
  interplanetary Bz (their size parameter Theta is regressed against -Bz).
  No solar-wind Bz measurements exist for Feb 1958 (first IMF spacecraft
  data begin ~1963), so that method cannot be applied to this historical
  event. Instead we use Starkov (1994a, "Mathematical model of the auroral
  boundaries", Geomagnetism & Aeronomy 34(3):331-336), which regresses the
  poleward, equatorward, and diffuse-aurora equatorward boundaries directly
  against the AL index. This *is* the standard "AL -> oval" method and is
  exactly what the digitized AU/AL time series can drive without additional
  unavailable inputs. Equations and coefficients below are transcribed from
  Sigernes et al. (2011), Eqs. (2)-(3) and Table 2 (which itself reproduces
  Starkov 1994a Table 2), not re-derived from Starkov's original (harder to
  access) paper -- flagged so the user can cross-check against the primary
  source before submission.
"""
import csv
import datetime as dt
import math

import aacgmv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import sys
sys.path.insert(0, "/sessions/vigilant-elegant-ptolemy/mnt/.claude/skills/scubas-figure-style/scripts")
from scubas_style import apply_style, PALETTE

apply_style()

# ---------------------------------------------------------------------
# 1. Load digitized AU/AL, baseline-correct
# ---------------------------------------------------------------------

def load(fn):
    dates, vals = [], []
    with open(fn) as f:
        r = csv.DictReader(f)
        for row in r:
            dates.append(float(row["Date"]))
            vals.append(float(row["H"]))
    pairs = sorted(zip(dates, vals))
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])

d_au, v_au = load("/sessions/vigilant-elegant-ptolemy/mnt/TAT-1/AU.csv")
d_al, v_al = load("/sessions/vigilant-elegant-ptolemy/mnt/TAT-1/AL.csv")

AU0 = v_au[:3].mean()
AL0 = v_al[:3].mean()


def nearest_dev(d, v, t0, base):
    i = np.argmin(np.abs(d - t0))
    return d[i], v[i] - base


SNAPSHOTS = [
    ("Pre-storm (quiet)", 0.51),
    ("SSC onset", 1.42),
    ("Peak (02:02 UT)", 2.042),
    ("Recovery", 4.05),
]

snap = []
for label, t0 in SNAPSHOTS:
    tA, vA = nearest_dev(d_au, v_au, t0, AU0)
    tL, vL = nearest_dev(d_al, v_al, t0, AL0)
    snap.append(dict(label=label, ut=tL, AU=vA, AL=vL))
    print(f"{label:22s} UT={tL:5.2f}  AU={vA:8.1f} nT  AL={vL:8.1f} nT")

# ---------------------------------------------------------------------
# 2. Starkov (1994) / Sigernes et al. (2011) AL-based oval model
# ---------------------------------------------------------------------

COEFF = {
    "poleward": {
        "A0": [-0.07, 24.54, -12.53, 2.15],
        "A1": [-10.06, 19.83, -9.33, 1.24],
        "A2": [-4.44, 7.47, -3.01, 0.25],
        "A3": [-3.77, 7.90, -4.73, 0.91],
        "a1": [-6.61, 10.17, -5.80, 1.19],
        "a2": [6.37, -1.10, 0.34, -0.38],
        "a3": [-4.48, 10.16, -5.87, 0.98],
    },
    "equatorward": {
        "A0": [1.61, 23.21, -10.97, 2.03],
        "A1": [-9.59, 17.78, -7.20, 0.96],
        "A2": [-12.07, 17.49, -7.96, 1.15],
        "A3": [-6.56, 11.44, -6.73, 1.31],
        "a1": [-2.22, 1.50, -0.58, 0.08],
        "a2": [-23.98, 42.79, -26.96, 5.56],
        "a3": [-20.07, 36.67, -24.20, 5.11],
    },
    "diffuse": {
        "A0": [3.44, 29.77, -16.38, 3.35],
        "A1": [-2.41, 7.89, -4.32, 0.87],
        "A2": [-0.74, 3.94, -3.09, 0.72],
        "A3": [-2.12, 3.24, -1.67, 0.31],
        "a1": [-1.68, -2.48, 1.58, -0.28],
        "a2": [8.69, -20.73, 13.03, -2.14],
        "a3": [8.61, -5.34, -1.36, 0.76],
    },
}


def poly3(b, x):
    b0, b1, b2, b3 = b
    return b0 + b1 * x + b2 * x**2 + b3 * x**3


def oval_boundary(boundary_name, AL_nT, t_hours):
    x = math.log10(abs(AL_nT)) if AL_nT != 0 else math.log10(1.0)
    c = COEFF[boundary_name]
    A0 = poly3(c["A0"], x)
    A1 = poly3(c["A1"], x)
    A2 = poly3(c["A2"], x)
    A3 = poly3(c["A3"], x)
    a1 = poly3(c["a1"], x)
    a2 = poly3(c["a2"], x)
    a3 = poly3(c["a3"], x)
    t = np.asarray(t_hours)
    theta = (
        A0
        + A1 * np.cos(np.radians(15 * (t - a1)))
        + A2 * np.cos(np.radians(15 * 2 * (t - a2)))
        + A3 * np.cos(np.radians(15 * 3 * (t - a3)))
    )
    return theta  # colatitude, degrees


t_grid = np.linspace(0, 24, 361)

for s in snap:
    for b in ("poleward", "equatorward", "diffuse"):
        s[b] = oval_boundary(b, s["AL"], t_grid)

# ---------------------------------------------------------------------
# 3. TAT-1 cable route in AACGM coordinates, per snapshot UT
# ---------------------------------------------------------------------

CABLE_WEST = (48.15, -54.13)  # Clarenville, Newfoundland
CABLE_EAST = (56.40, -5.47)   # Oban, Scotland


def slerp_route(p1, p2, n=25):
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    x1 = np.array([math.cos(lat1) * math.cos(lon1), math.cos(lat1) * math.sin(lon1), math.sin(lat1)])
    x2 = np.array([math.cos(lat2) * math.cos(lon2), math.cos(lat2) * math.sin(lon2), math.sin(lat2)])
    omega = math.acos(np.clip(np.dot(x1, x2), -1, 1))
    pts = []
    for f in np.linspace(0, 1, n):
        if omega < 1e-6:
            xv = x1
        else:
            xv = (math.sin((1 - f) * omega) * x1 + math.sin(f * omega) * x2) / math.sin(omega)
        lat = math.degrees(math.asin(np.clip(xv[2], -1, 1)))
        lon = math.degrees(math.atan2(xv[1], xv[0]))
        pts.append((lat, lon))
    return pts


route_geo = slerp_route(CABLE_WEST, CABLE_EAST, n=25)

for s in snap:
    hh = int(s["ut"])
    mm = int(round((s["ut"] - hh) * 60))
    dtime = dt.datetime(1958, 2, 11, hh, mm)
    route_mlat, route_mlt = [], []
    for lat, lon in route_geo:
        mlat, mlon, mlt = aacgmv2.get_aacgm_coord(lat, lon, 0, dtime)
        route_mlat.append(mlat)
        route_mlt.append(mlt)
    s["route_colat"] = 90 - np.array(route_mlat)
    s["route_mlt"] = np.array(route_mlt)
    wlat, wlon, wmlt = aacgmv2.get_aacgm_coord(CABLE_WEST[0], CABLE_WEST[1], 0, dtime)
    elat, elon, emlt = aacgmv2.get_aacgm_coord(CABLE_EAST[0], CABLE_EAST[1], 0, dtime)
    s["west_colat"], s["west_mlt"] = 90 - wlat, wmlt
    s["east_colat"], s["east_mlt"] = 90 - elat, emlt

# ---------------------------------------------------------------------
# 4. Plot: 4-panel polar small multiples
# ---------------------------------------------------------------------

BOUND_STYLE = {
    "poleward": dict(color=PALETTE["deep_blue"], ls="-", lw=1.8, label="Poleward boundary"),
    "equatorward": dict(color=PALETTE["deep_blue"], ls="--", lw=1.8, label="Equatorward boundary"),
    "diffuse": dict(color=PALETTE["accent_grey"], ls=":", lw=1.4, label="Diffuse aurora (equatorward)"),
}

fig, axes = plt.subplots(1, 4, subplot_kw={"projection": "polar"}, figsize=(12.4, 3.3))
fig.subplots_adjust(wspace=0.75)

for ax, s in zip(axes, snap):
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 40)
    ax.set_rticks([10, 20, 30, 40])
    ax.set_rlabel_position(135)
    ax.set_yticklabels(["80°", "70°", "60°", "50°"], fontsize=6.5)
    ax.set_xticks(np.radians(np.arange(0, 360, 90)))
    ax.set_xticklabels(["00", "06", "12", "18"], fontsize=7.5)
    ax.grid(color=PALETTE["light_grey"], linewidth=0.6)
    ax.spines["polar"].set_color(PALETTE["accent_grey"])

    theta_grid = np.radians(t_grid * 15)
    for b in ("diffuse", "equatorward", "poleward"):
        style = dict(BOUND_STYLE[b])
        lbl = style.pop("label")
        ax.plot(theta_grid, s[b], **style, label=lbl)

    ax.fill_between(theta_grid, s["poleward"], s["equatorward"],
                     color=PALETTE["golden_yellow"], alpha=0.35, linewidth=0)

    ax.plot(np.radians(s["route_mlt"] * 15), s["route_colat"],
             color=PALETTE["accent_red"], lw=2.2, zorder=5, label="TAT-1 cable route")
    ax.scatter([np.radians(s["west_mlt"] * 15)], [s["west_colat"]],
               marker="s", s=22, color=PALETTE["accent_red"], zorder=6)
    ax.scatter([np.radians(s["east_mlt"] * 15)], [s["east_colat"]],
               marker="o", s=22, color=PALETTE["accent_red"], zorder=6)

    hh = int(s["ut"]); mm = int(round((s["ut"] - hh) * 60))
    flag = "*" if abs(s["AL"]) > 800 else ""
    ax.set_title(f"{s['label']}\n{hh:02d}:{mm:02d} UT   AL = {s['AL']:.0f} nT{flag}",
                 fontsize=8.5, pad=14)

handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.06), fontsize=8, frameon=False)

fig.suptitle("Northern auroral oval expansion during the Feb 11, 1958 storm (Starkov 1994 AL-based model)\n"
             "Corrected geomagnetic coordinates (AACGM-v2, epoch 1958.1); square = Clarenville (west end), circle = Oban (east end)",
             fontsize=9.2, y=1.14)

peak_AL = [s["AL"] for s in snap if s["label"].startswith("Peak")][0]
fig.text(0.5, -0.14,
         f"* Starkov (1994) boundary regression was fit over an AL range of order $\\pm$800 nT (Sigernes et al., 2011); "
         f"the peak-panel |AL| = {abs(peak_AL):.0f} nT lies outside this range and the equatorward boundary there should be treated as an\n"
         "extrapolation, not a directly validated result. Peak snapshot taken at 02:02-02:03 UT (the sustained post-spike plateau matching the cable's "
         "own reported voltage peak time, Tapley in Weaver et al. 1959), not the brief single-sample -1476 nT dip at 01:38 UT.\n"
         "AL/AU baseline-corrected against the mean of the first 3 digitized points (UT < 0.1 h), assumed pre-storm quiet.",
         ha="center", fontsize=6.8, color=PALETTE["accent_grey"])

fig.savefig("/sessions/vigilant-elegant-ptolemy/mnt/outputs/oval/auroral_oval_1958_storm.png",
            dpi=600, bbox_inches="tight")
fig.savefig("/sessions/vigilant-elegant-ptolemy/mnt/outputs/oval/auroral_oval_1958_storm.pdf",
            bbox_inches="tight")
print("saved figure")
