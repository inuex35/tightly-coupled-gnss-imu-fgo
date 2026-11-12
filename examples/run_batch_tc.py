"""Batch (whole-run) tightly-coupled GNSS RTK + IMU on one drive.

Usage:
  python examples/run_batch_tc.py rover.obs base.obs base.nav imu.csv [reference.csv]

Env (besides the BatchConfig ones, see src/gnss_fgo/batch/config.py):
  BASE_LLH   base antenna "lat,lon,h" [deg, deg, m]; default base.obs header.
             PPC-Dataset README: tokyo 35.66633426,139.79220181,59.82,
             nagoya 35.13470947,136.97757427,104.718
  LEVER_ARM  antenna in body FLU [m]; tokyo 0.31,0,0.55, nagoya 0.593,0.670,1.216
  CACHE      .npz of the front-end arrays: read if present, else written
  MAX_EP     process only the first MAX_EP rover epochs
  OUT_NPZ    write trajectories, FIX flags and integer relations here

The LM solves run on gtsam.cuda (cuDSS) when available, else on the CPU.
Build the C LAMBDA once for speed: python -c "from gnss_fgo.batch import lambda_c; lambda_c.build()"
"""
import os
import sys

import numpy as np

from gnss_fgo.batch import BatchConfig, load_imu_csv, run_batch
from gnss_fgo.batch import frontend


def base_ecef_from_env():
    if "BASE_LLH" not in os.environ:
        return None
    from cssrlib.gnss import pos2ecef
    lat, lon, h = (float(v) for v in os.environ["BASE_LLH"].split(","))
    return pos2ecef(np.array([np.deg2rad(lat), np.deg2rad(lon), h]))


def reference_ecef(path, tow):
    """reference.csv ECEF (antenna phase centre) at the rover epochs; NaN if missing."""
    ref = np.loadtxt(path, delimiter=",", skiprows=1)
    idx = {round(t, 1): i for i, t in enumerate(ref[:, 0])}
    sel = np.array([idx.get(round(t, 1), -1) for t in tow])
    out = np.full((len(tow), 3), np.nan)
    out[sel >= 0] = ref[sel[sel >= 0], 5:8]
    return out


def summary(name, est, truth, mask=None):
    e = np.linalg.norm(est - truth, axis=1)
    if mask is not None:
        e = e[mask]
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return f"{name:10s}: no epochs"
    return (f"{name:10s}: n={len(e):6d} RMS {np.sqrt(np.mean(e ** 2)):7.3f} m  "
            f"median {np.median(e):6.3f} m  <0.5 m {100 * np.mean(e < 0.5):5.1f}%  "
            f"<3 m {100 * np.mean(e < 3):5.1f}%  max {e.max():7.2f} m")


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    rover, base, nav, imu_csv = sys.argv[1:5]
    ref_csv = sys.argv[5] if len(sys.argv) > 5 else None
    cache = os.environ.get("CACHE")
    max_ep = int(os.environ["MAX_EP"]) if "MAX_EP" in os.environ else None

    if cache and os.path.exists(cache):
        meas = frontend.load(cache)
        print(f"front end: loaded {cache}")
    else:
        meas = frontend.prepare(rover, base, nav, base_ecef_from_env(), max_ep)
        if cache:
            frontend.save(cache, meas)
    if max_ep and len(meas["tow"]) > max_ep:
        meas = dict(meas, tow=meas["tow"][:max_ep],
                    rows=meas["rows"][meas["rows"][:, 0] < max_ep],
                    dop=meas["dop"][meas["dop"][:, 0] < max_ep])

    cfg = BatchConfig()
    print(f"lever arm (FLU) {cfg.lever}, solver {cfg.solver}")
    res = run_batch(meas, load_imu_csv(imu_csv), cfg)

    if ref_csv:
        truth = reference_ecef(ref_csv, res["tow"])
        print(summary("GNSS-only", res["gnss_fix"], truth))
        print(summary("TC float", res["tc_float"], truth))
        print(summary("TC fix", res["tc_fix"], truth))
        print(summary("  FIX", res["tc_fix"], truth, res["fix"]))
        print(summary("  not FIX", res["tc_fix"], truth, ~res["fix"]))
        res["truth"] = truth
    if "OUT_NPZ" in os.environ:
        np.savez(os.environ["OUT_NPZ"], **{k: v for k, v in res.items() if k != "stats"})
        print(f"saved {os.environ['OUT_NPZ']}")


if __name__ == "__main__":
    main()
