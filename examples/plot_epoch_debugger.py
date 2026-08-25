#!/usr/bin/env python3
"""Per-epoch FGO internal-state debugger.

Consumes the archives written by ``run_imu_gnss_tc.py``:
  - ``.npz``  (SAVE_NPZ=1): dense per-epoch scalar arrays (auto-discovered keys)
  - pickle    (SAVE_PER_SAT=1): per-satellite residuals / elevation / SNR

Panels share the epoch axis so AR decisions, quality gates, recovery events,
and residual spikes can be correlated at a glance.

Usage:
    python examples/plot_epoch_debugger.py results/c28_full.npz \
        [--per-sat results/c28_per_sat.pkl] [--output debug.png] \
        [--start 0] [--end 5000] [--show]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
import numpy as np

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIX_COLOR = "#1a9850"
FLT_COLOR = "#fdae61"

# Raster event rows: label -> list of info keys, any of which marks an epoch.
EVENT_ROWS = OrderedDict(
    [
        ("AR reject", ["ar_ddpr_xvalidate_reject", "ar_context_reject",
                       "low_nb_fix_reject", "lambda_corr_hard_reject"]),
        ("FDE", ["fde_reject", "fde_skipped", "doppler_fde"]),
        ("DDPR ladder", ["ddpr_bad", "ddpr_fast_recover", "ddpr_recover"]),
        ("CP hold", None),  # filled dynamically from cp_hold_* keys
        ("Sanity/recovery", ["sanity_pose_gap", "sanity_pose_replaced",
                             "sanity_dd_removed", "gnss_skip", "error"]),
        ("Slips/ZUPT", ["n_slip", "zupt", "pim_discontinuity"]),
    ]
)


def _col(npz: dict, key: str):
    v = npz.get(key)
    return np.asarray(v, dtype=float) if v is not None else None


def _mask_any(npz: dict, keys):
    m = np.zeros(len(next(iter(npz.values()))), dtype=bool) if npz else None
    for k in keys:
        v = _col(npz, k)
        if v is not None:
            m |= ~np.isnan(v) & (v != 0)
    return m


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def dynamic_cp_hold_keys(npz: dict):
    return sorted(k for k in npz if k.startswith("cp_hold_")
                  and not k.endswith("retrigger_streak"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--per-sat", type=Path, default=None)
    ap.add_argument("--output", "-o", type=Path, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    npz = load_npz(args.npz)
    per_sat = None
    if args.per_sat and args.per_sat.exists():
        with open(args.per_sat, "rb") as fh:
            per_sat = pickle.load(fh)

    lo, hi = args.start, args.end
    sl = slice(lo, hi)

    def c(key):
        return _col(npz, key)

    err = c("err3d")
    smode = c("smode")
    nb = c("nb")
    tow = c("tow") if "tow" in npz else np.arange(len(err))
    fix = (smode == 4) if smode is not None else ((nb > 0) if nb is not None else None)

    EVENT_ROWS["CP hold"] = dynamic_cp_hold_keys(npz)

    n_panels = 5 + len(EVENT_ROWS)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 2.0 * n_panels), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.6, 1.4, 1.2, 1.2] + [0.55] * len(EVENT_ROWS)})
    ax = axes[0]

    # Panel 1: error timeline
    ax.plot(tow[sl], err[sl], color="#333333", lw=0.7, alpha=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("3D err [m]")
    ax.set_title(f"FGO epoch debugger — {args.npz.name}")
    if fix is not None:
        ax.scatter(tow[sl][fix[sl]], err[sl][fix[sl]], s=4, c=FIX_COLOR,
                   label="FIX", zorder=3)
        ax.scatter(tow[sl][~fix[sl]], err[sl][~fix[sl]], s=4, c=FLT_COLOR,
                   label="FLT", zorder=3)
        ax.legend(loc="upper right", fontsize=7)

    # Panel 2: DDPR residuals
    ax = axes[1]
    for key, col in (("main_ddpr_res", "#1f77b4"),
                     ("ar_context_reject_main_ddpr_res", "#d62728"),
                     ("ddpr_res_at_pred", "#9467bd")):
        v = c(key)
        if v is not None:
            ax.plot(tow[sl], v[sl], lw=0.7, color=col, alpha=0.85, label=key)
            bad = ~np.isnan(v)
            ax.scatter(tow[bad][sl], v[bad][sl], s=3, color=col)
    ax.set_ylabel("DDPR res [m]")
    ax.legend(loc="upper right", fontsize=6)

    # Panel 3: ambiguity count + AR outcome code
    ax = axes[2]
    if nb is not None:
        ax.step(tow[sl], nb[sl], where="post", lw=0.7, color="#2ca02c")
        ax.set_ylabel("fixed amb nb")
    ax2 = ax.twinx()
    oc = c("ar_outcome_code")
    if oc is not None:
        ax2.step(tow[sl], oc[sl], where="post", lw=0.8, color="#7f7f7f", alpha=0.9)
        ax2.set_ylabel("AR outcome code", color="#7f7f7f")
        codes = sorted(set(int(x) for x in oc[~np.isnan(oc)]))
        ax2.set_yticks(codes)
        ax2.set_yticklabels([str(x) for x in codes], fontsize=6)

    # Panel 4: geometry quality
    ax = axes[3]
    gdop, nsat = c("gdop"), c("nsat")
    if nsat is not None:
        ax.step(tow[sl], nsat[sl], where="post", lw=0.7, color="#1f77b4")
        ax.set_ylabel("nsat")
    if gdop is not None:
        ax2b = ax.twinx()
        ax2b.plot(tow[sl], gdop[sl], lw=0.7, color="#ff7f0e")
        ax2b.set_ylabel("GDOP", color="#ff7f0e")

    # Panel 5: innovation / lambda correction
    ax = axes[4]
    for key, col in (("innovation", "#8c564b"), ("lambda_correction", "#e377c2")):
        v = c(key)
        if v is not None:
            ax.plot(tow[sl], v[sl], lw=0.7, color=col, alpha=0.85, label=key)
    ax.set_ylabel("innov / λcorr")
    ax.legend(loc="upper right", fontsize=6)

    # Event rasters
    for i, (label, keys) in enumerate(EVENT_ROWS.items()):
        ax = axes[5 + i]
        m = _mask_any(npz, keys) if keys else np.zeros(len(err), dtype=bool)
        ax.imshow(m[sl][None, :], aspect="auto", interpolation="nearest",
                  cmap="Greys", vmin=0, vmax=1,
                  extent=(tow[sl][0] if len(tow[sl]) else 0,
                          tow[sl][-1] if len(tow[sl]) else 1, 0, 1))
        ax.set_yticks([])
        active = int(m[sl].sum())
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)
        ax.text(1.002, 0.5, f"{active}", transform=ax.transAxes,
                fontsize=6, va="center", color="#888888")
    axes[-1].set_xlabel("GPS TOW [s]" if "tow" in npz else "epoch")

    # Per-satellite residual heatmap (optional, appended figure)
    if per_sat and per_sat.get("per_sat"):
        fig2, axh = plt.subplots(figsize=(14, 4))
        sats = sorted({s for d in per_sat["per_sat"] if d for s in d})
        mat = np.full((len(sats), len(per_sat["per_sat"])), np.nan)
        sidx = {s: i for i, s in enumerate(sats)}
        for j, d in enumerate(per_sat["per_sat"]):
            if d:
                for s, r in d.items():
                    mat[sidx[s], j] = r
        im = axh.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
        axh.set_yticks(range(len(sats)))
        axh.set_yticklabels([str(s) for s in sats], fontsize=5)
        axh.set_xlabel("epoch")
        axh.set_title("main DDPR residual per satellite [m]")
        fig2.colorbar(im, ax=axh, shrink=0.8)
        out_base = args.output or args.npz.with_name(args.npz.stem + "_debug.png")
        out2 = out_base.with_name(out_base.stem + "_persat.png")
        fig2.savefig(out2, dpi=140, bbox_inches="tight")
        print(f"wrote {out2}")

    out = args.output or args.npz.with_name(args.npz.stem + "_debug.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
