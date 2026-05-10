#!/usr/bin/env python3
"""Render docs/tokyo_defaults.png — tokyo run1/run2/run3 planar trajectory.

Usage:
  python examples/plot_tokyo_defaults.py \
      run1.npz run2.npz run3.npz [-o docs/tokyo_defaults.png] [--clip 0.5]
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def ecef_to_lla(x, y, z):
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - e2 * n / (n + h)))
    return lat, lon, h


def to_enu(xyz, base):
    lat0, lon0, _ = ecef_to_lla(*base)
    sl, cl = np.sin(lat0), np.cos(lat0)
    so, co = np.sin(lon0), np.cos(lon0)
    r = np.array([[-so, co, 0.0],
                  [-sl * co, -sl * so, cl],
                  [cl * co, cl * so, sl]])
    return (xyz - base) @ r.T


def panel_metrics(d):
    err = d['err3d']
    smode = d['smode']
    fix_mask = smode == 4
    n = len(err)
    nfix = int(fix_mask.sum())
    all_rms = float(np.sqrt(np.mean(err ** 2)))
    fix_rms = (float(np.sqrt(np.mean(err[fix_mask] ** 2)))
               if nfix > 0 else float('nan'))
    fix_pct = 100.0 * nfix / n if n > 0 else 0.0
    return n, all_rms, fix_rms, fix_pct


def draw_panel(ax, d, clip_max, cmap):
    base = d['base_xyz']
    truth_enu = to_enu(d['truth_xyz'], base)
    sol_enu = to_enu(d['sol_xyz'], base)
    err = d['err3d']

    ax.plot(truth_enu[:, 0], truth_enu[:, 1],
            color='black', lw=1.0, alpha=0.5, label='truth')
    ax.plot(sol_enu[:, 0], sol_enu[:, 1],
            color='tab:blue', lw=0.6, alpha=0.4)
    sc = ax.scatter(sol_enu[:, 0], sol_enu[:, 1],
                    c=np.clip(err, 0.0, clip_max),
                    cmap=cmap, s=6, alpha=0.85,
                    vmin=0.0, vmax=clip_max)
    ax.set_aspect('equal')
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.grid(True, alpha=0.3)
    return sc


def draw_fix_status_panel(ax, d):
    """Planar plot colored by FIX status (smode==4 → FIX, else FLT)."""
    base = d['base_xyz']
    truth_enu = to_enu(d['truth_xyz'], base)
    sol_enu = to_enu(d['sol_xyz'], base)
    smode = d['smode']
    fix_mask = (smode == 4)

    ax.plot(truth_enu[:, 0], truth_enu[:, 1],
            color='black', lw=1.0, alpha=0.5, label='truth')
    ax.plot(sol_enu[:, 0], sol_enu[:, 1],
            color='lightgray', lw=0.6, alpha=0.4, zorder=1)
    if np.any(~fix_mask):
        ax.scatter(sol_enu[~fix_mask, 0], sol_enu[~fix_mask, 1],
                   c='tab:red', s=4, alpha=0.4, label='FLT', zorder=2)
    if np.any(fix_mask):
        ax.scatter(sol_enu[fix_mask, 0], sol_enu[fix_mask, 1],
                   c='tab:green', s=4, alpha=0.7, label='FIX', zorder=3)
    ax.set_aspect('equal')
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, markerscale=2.0, framealpha=0.85)


def main():
    args = sys.argv[1:]
    out = 'docs/tokyo_defaults.png'
    clip_max = 0.5
    files = []
    i = 0
    while i < len(args):
        if args[i] == '-o':
            out = args[i + 1]
            i += 2
        elif args[i] == '--clip':
            clip_max = float(args[i + 1])
            i += 2
        else:
            files.append(args[i])
            i += 1

    if len(files) != 3:
        print("Usage: plot_tokyo_defaults.py run1.npz run2.npz run3.npz "
              "[-o out.png] [--clip 0.5]")
        sys.exit(1)

    cmap = 'RdYlBu_r'
    fig = plt.figure(figsize=(15, 10.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.05],
                          hspace=0.30, wspace=0.25)

    sc = None
    for col, npz_path in enumerate(files):
        d = np.load(npz_path)
        n, all_rms, fix_rms, fix_pct = panel_metrics(d)

        ax_top = fig.add_subplot(gs[0, col])
        draw_fix_status_panel(ax_top, d)
        ax_top.set_title(
            f'run{col + 1}  ({n} ep)\n'
            f'AllRMS={all_rms:.2f} m   FixRMS={fix_rms:.3f} m   '
            f'fix={fix_pct:.1f}%'
        )

        ax = fig.add_subplot(gs[1, col])
        sc = draw_panel(ax, d, clip_max, cmap)

    cax = fig.add_subplot(gs[2, :])
    fig.colorbar(sc, cax=cax, orientation='horizontal',
                 label=f'3D error [m] (clip {clip_max:g} m)')

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
