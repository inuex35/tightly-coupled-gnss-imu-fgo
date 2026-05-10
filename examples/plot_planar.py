#!/usr/bin/env python3
"""Top-down ENU plot of solved vs truth trajectory.

Highlights:
  * solved trajectory coloured by err3d (capped 15 m)
  * truth path overlaid in light grey
  * wrong-basin FIX onsets marked
  * dominant bad-run start points circled

Usage:
  python plot_planar.py [npz] [out.png]
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OK_THRESH = 0.10  # 10 cm

npz = sys.argv[1] if len(sys.argv) > 1 else \
    "results/baseline_tokyo2_2000ep.npz"
out = sys.argv[2] if len(sys.argv) > 2 else \
    "results/wrong_basin_planar.png"

d = np.load(npz, allow_pickle=True)
err = d["err3d"]
nb = d["nb"]
ph = d["phase"]
enu = d["enu"]
base_xyz = d["base_xyz"]
sol_xyz = d["sol_xyz"]
truth_xyz = d["truth_xyz"]
fix = nb > 0
OK = err <= OK_THRESH
P2 = ph == 2

# ECEF -> ENU using base position. Derive R from sol-vs-enu since
# both are saved.  enu = R @ (sol_xyz - base_xyz)  for each epoch.
def ecef2enu_R(base_xyz):
    """Return R (3x3) rotating ECEF deltas into ENU at base lat/lon."""
    x, y, z = base_xyz
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1 - e2))
    # iterate (closed-form is fine; one pass is plenty for plotting)
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    R = np.array([[-so, co, 0],
                  [-sl * co, -sl * so, cl],
                  [cl * co, cl * so, sl]])
    return R

R = ecef2enu_R(base_xyz)
truth_enu = (truth_xyz - base_xyz) @ R.T  # apply R to each row
sol_enu = (sol_xyz - base_xyz) @ R.T

# Bad segment detection (continuous err > 10 cm in P2)
in_bad = False
bad_segs = []
start = None
for i in range(len(err)):
    if P2[i] and not OK[i] and not in_bad:
        in_bad = True
        start = i
    elif (not P2[i] or OK[i]) and in_bad:
        bad_segs.append((start, i - 1, i - start))
        in_bad = False
if in_bad:
    bad_segs.append((start, len(err) - 1, len(err) - start))
top_segs = sorted(bad_segs, key=lambda r: -r[2])[:5]

# OK -> FIX & wrong-basin onsets
WB_fix = fix & ~OK & P2
wb_starts = []
prev_ok = True
for i in range(len(err)):
    if not P2[i]:
        prev_ok = True
        continue
    if WB_fix[i] and prev_ok:
        wb_starts.append(i)
    prev_ok = OK[i]

fig, ax = plt.subplots(figsize=(12, 10))

# Truth path in light grey
ax.plot(truth_enu[:, 0], truth_enu[:, 1], color="0.7", lw=1.2,
        label="truth", zorder=1)

# Solved trajectory coloured by err3d
VMAX = 0.5
sc = ax.scatter(sol_enu[:, 0], sol_enu[:, 1],
                 c=np.minimum(err, VMAX), s=8, cmap="turbo",
                 vmin=0, vmax=VMAX, zorder=2)

# Phase 1 start
ax.plot(sol_enu[0, 0], sol_enu[0, 1], "k^", ms=10,
        label="ep 0 start", zorder=4)
# Phase 1 -> 2 transition
p1_end = int(np.where(ph == 1)[0][-1])
ax.plot(sol_enu[p1_end, 0], sol_enu[p1_end, 1], "ks", ms=10,
        label=f"ep {p1_end} P1->P2", zorder=4)

# Wrong-basin FIX onsets (small black ×)
for s in wb_starts:
    ax.plot(sol_enu[s, 0], sol_enu[s, 1], "kx", ms=8, mew=1.5,
            zorder=5)

# Top-5 bad-run starts (red circle + label)
for s, e, n in top_segs:
    ax.plot(sol_enu[s, 0], sol_enu[s, 1], "o", ms=16, mfc="none",
            mec="red", mew=2.0, zorder=6)
    ax.annotate(f"ep {s}\nbad len={n}\nerr mean={err[s:e+1].mean():.1f} m",
                (sol_enu[s, 0], sol_enu[s, 1]),
                fontsize=9, color="red",
                xytext=(10, 10), textcoords="offset points",
                zorder=6)
    # Also mark end with smaller red x
    ax.plot(sol_enu[e, 0], sol_enu[e, 1], "x", color="red", ms=10, mew=2,
            zorder=6)

ax.set_aspect("equal")
ax.set_xlabel("E [m]  (rel. to base)")
ax.set_ylabel("N [m]  (rel. to base)")
ax.set_title(
    f"{npz}\n"
    f"Solved (colour = err3d, capped {VMAX} m) vs truth (grey).  "
    f"Black × = OK→wrong-basin FIX onset, "
    f"red ○ = top-5 bad-run starts."
)
ax.legend(loc="lower right", fontsize=9)
plt.colorbar(sc, ax=ax, label="err3d [m]")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out, dpi=150)
print(f"saved: {out}")
print(f"truth_enu range  E [{truth_enu[:,0].min():.1f}, {truth_enu[:,0].max():.1f}]"
      f"  N [{truth_enu[:,1].min():.1f}, {truth_enu[:,1].max():.1f}]")
print(f"sol_enu   range  E [{sol_enu[:,0].min():.1f}, {sol_enu[:,0].max():.1f}]"
      f"  N [{sol_enu[:,1].min():.1f}, {sol_enu[:,1].max():.1f}]")
print(f"\nTop-5 bad-run starts:")
for s, e, n in top_segs:
    print(f"  ep {s:4d}..{e:4d}  start ENU=({sol_enu[s,0]:7.1f}, {sol_enu[s,1]:7.1f})"
          f"  truth=({truth_enu[s,0]:7.1f}, {truth_enu[s,1]:7.1f})"
          f"  err mean={err[s:e+1].mean():5.2f} m  max={err[s:e+1].max():5.2f} m")
