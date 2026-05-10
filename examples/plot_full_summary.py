#!/usr/bin/env python3
"""Three-panel diagnostic for a TC run:
  top    err3d vs epoch (log y) coloured by FIX/FLT
  middle nb (LAMBDA fix count) + smode  vs epoch
  bottom ENU trajectory coloured by err (capped 0.1 m for FIX,
                                         capped 30 m for FLT)
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

npz = sys.argv[1] if len(sys.argv) > 1 else "results/c28_full.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "results/full_summary.png"

d = np.load(npz, allow_pickle=True)
err = d["err3d"]
nb = d["nb"]
ph = d["phase"]
sm = d["smode"]
base_xyz = d["base_xyz"]
sol_xyz = d["sol_xyz"]
truth_xyz = d["truth_xyz"]
fix = nb > 0
P2 = ph == 2
N = len(err)


def ecef2enu_R(base):
    x, y, z = base
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        Nn = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * Nn * np.sin(lat), p)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


R = ecef2enu_R(base_xyz)
truth_enu = (truth_xyz - base_xyz) @ R.T
sol_enu = (sol_xyz - base_xyz) @ R.T

ep = np.arange(N)
fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 0.8, 2.2], hspace=0.30)
ax_t = fig.add_subplot(gs[0])
ax_n = fig.add_subplot(gs[1], sharex=ax_t)
ax_xy = fig.add_subplot(gs[2])

# Panel 1: err3d vs epoch, log y
e_clip = np.maximum(err, 1e-3)
fix2 = fix & P2
flt2 = ~fix & P2
ax_t.semilogy(ep[ph == 1], e_clip[ph == 1], '.', ms=2, color='gray',
              label=f'P1 ({(ph == 1).sum()})')
ax_t.semilogy(ep[fix2], e_clip[fix2], '.', ms=3, color='tab:blue',
              label=f'P2 FIX ({fix2.sum()})')
ax_t.semilogy(ep[flt2], e_clip[flt2], '.', ms=2, color='tab:red',
              label=f'P2 FLT ({flt2.sum()})')
ax_t.axhline(0.10, color='k', ls='--', lw=0.6, alpha=0.5)
ax_t.text(0, 0.11, '10 cm', fontsize=8)
ax_t.set_ylabel('err3d [m] (log)')
ax_t.set_title(
    f'{npz}: '
    f'FIX {fix2.sum()} ({100*fix2.sum()/P2.sum():.1f}%) — '
    f'med {np.median(err[fix2])*100:.1f} cm, RMS {np.sqrt((err[fix2]**2).mean())*100:.1f} cm.  '
    f'FLT {flt2.sum()} ({100*flt2.sum()/P2.sum():.1f}%) — '
    f'med {np.median(err[flt2]):.2f} m, RMS {np.sqrt((err[flt2]**2).mean()):.1f} m'
)
ax_t.legend(loc='upper left', fontsize=8, ncol=3)
ax_t.set_ylim(1e-3, 5e2)
ax_t.grid(True, alpha=0.3, which='both')

# Panel 2: nb (left axis) + smode (right axis)
ax_n.plot(ep, nb, color='tab:blue', lw=0.5, label='nb (LAMBDA fix sats)')
ax_n.set_ylabel('nb', color='tab:blue')
ax_n.tick_params(axis='y', labelcolor='tab:blue')
ax_n.set_xlabel('epoch')
ax_n.grid(True, alpha=0.3)
ax2 = ax_n.twinx()
ax2.plot(ep, sm, color='tab:orange', lw=0.5, alpha=0.6, label='smode')
ax2.set_ylabel('smode', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Panel 3: ENU trajectory
ax_xy.plot(truth_enu[:, 0], truth_enu[:, 1], color='0.85', lw=1.0, zorder=1)
# FLT in red gradient (cap 30 m)
ax_xy.scatter(sol_enu[flt2, 0], sol_enu[flt2, 1],
              c=np.minimum(err[flt2], 30), s=3,
              cmap='autumn_r', vmin=0, vmax=30,
              alpha=0.6, zorder=2, label=f'FLT (n={flt2.sum()})')
# FIX overlaid
sc = ax_xy.scatter(sol_enu[fix2, 0], sol_enu[fix2, 1],
                    c=np.minimum(err[fix2], 0.1), s=10,
                    cmap='turbo', vmin=0, vmax=0.1,
                    zorder=4, edgecolors='none',
                    label=f'FIX (n={fix2.sum()})')
ax_xy.plot(sol_enu[0, 0], sol_enu[0, 1], 'k^', ms=8,
           label='ep 0', zorder=5)
ax_xy.set_aspect('equal')
ax_xy.set_xlabel('E [m]'); ax_xy.set_ylabel('N [m]')
ax_xy.set_title('Trajectory: FIX (turbo, ≤10 cm) over FLT (red gradient, cap 30 m), truth gray')
ax_xy.legend(loc='lower left', fontsize=8)
plt.colorbar(sc, ax=ax_xy, label='FIX err [m]', shrink=0.6, pad=0.02)
ax_xy.grid(True, alpha=0.3)

plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'saved: {out}')
