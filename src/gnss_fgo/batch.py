"""Whole-run batch pipeline: GNSS-only batch -> TC float -> FDE -> AR -> TC fix."""

import time

import numpy as np

from . import batch_ar as _ar
from . import batch_graph as _graph
from .batch_graph import load, prepare, save
from .config import BatchConfig
from .utils import load_imu_csv as _load_imu_csv


def load_imu_csv(path: str):
    """PPC-Dataset imu.csv -> (tow, acc [m/s^2], gyro [rad/s]); raw axes (FLU)."""
    return _load_imu_csv(path, as_arrays=True)


def run_batch(meas, imu, cfg=None, log=print):
    """Run the batch pipeline on front-end arrays (prepare).

    meas: dict(rows, dop, tow, rb, pos0); imu: (tow, acc [m/s^2], gyro [rad/s])
    in body FLU. Returns a dict with antenna ECEF trajectories (gnss_fix,
    tc_float, tc_fix), per-epoch FIX flags (fix, fix_rejected), the accepted
    integer relations and stats.
    """
    cfg = cfg or BatchConfig()
    rows_raw, tow, rb, pos0 = meas["rows"], meas["tow"], meas["rb"], meas["pos0"]
    n_ep = len(tow)

    def solve(graph, init):
        return _graph.solve(graph, init, cfg.solver, cfg.max_iters)

    out = {"tow": tow, "stats": {}}
    t_all = time.perf_counter()

    # --- GNSS-only batch: initial trajectory for the TC graph.
    t0 = time.perf_counter()
    rows = (_graph.tropo_correct(rows_raw, np.tile(pos0, (n_ep, 1)), rb)
            if cfg.tropo else rows_raw)
    g_graph, g_init = _graph.build_gnss_graph(rows, tow, rb, pos0, cfg)
    g_flt, _, _ = solve(g_graph, g_init)
    groups = _ar.group_epochs(rows)
    cov = _ar.arc_covariances(g_graph, g_flt, groups, n_ep, _graph.GNSS_DIMS)
    acc, _, _, st = _ar.resolve(groups, g_flt, cov, cfg.ratio, cfg.lambda_workers)
    g_fix, _, _ = solve(_graph.add_fix(g_graph, acc, cfg.sig_fix), g_flt)
    out["gnss_fix"] = _graph.gnss_positions(g_fix, n_ep)
    log(f"GNSS-only batch: {len(acc)} integer relations, "
        f"{time.perf_counter() - t0:.1f}s")

    # --- tightly coupled float.
    t0 = time.perf_counter()
    if cfg.tropo:
        rows = _graph.tropo_correct(rows_raw, out["gnss_fix"], rb)
    tc_meas = dict(meas, rows=rows)
    graph, init, ecef_T_nav, idx = _graph.build_tc_graph(tc_meas, imu, out["gnss_fix"], cfg)
    log(f"TC graph: {n_ep} keyframes, {graph.size()} factors, {init.size()} variables, "
        f"ZUPT on {idx['n_zupt']} epochs ({time.perf_counter() - t0:.1f}s)")
    flt, dt, info = solve(graph, init)
    log(f"TC float: {dt:.2f}s {info}")
    flt, keep_cp = _graph.fde(graph, flt, idx, rows, cfg, solve, log)
    out["tc_float"] = _graph.antenna_ecef(flt, ecef_T_nav, n_ep, cfg.lever)

    # --- ambiguity resolution on the surviving carrier rows.
    t0 = time.perf_counter()
    groups = _ar.group_epochs(rows[keep_cp])
    cov = _ar.arc_covariances(graph, flt, groups, n_ep, _graph.TC_DIMS)
    acc, sources, passed, st = _ar.resolve(groups, flt, cov, cfg.ratio, cfg.lambda_workers)
    log(f"AR: {st['ratio_pass']}/{n_ep} epochs pass ratio >= {cfg.ratio}, "
        f"{len(acc)} integer relations, {st['conflict']} conflicts "
        f"({time.perf_counter() - t0:.1f}s)")

    # --- tightly coupled fixed solve and per-epoch FIX validation.
    fgraph = _graph.add_fix(graph, acc, cfg.sig_fix)
    fix, dt, info = solve(fgraph, flt)
    log(f"TC fix: {dt:.2f}s {info}")
    _, cp_res, _ = _graph.residuals(fgraph, fix, idx)
    fix_ok, fix_rej, _ = _ar.validate_fix(rows, keep_cp, acc, cp_res, n_ep, cfg)
    out.update(tc_fix=_graph.antenna_ecef(fix, ecef_T_nav, n_ep, cfg.lever),
               fix=fix_ok, fix_rejected=fix_rej, accepted=np.array(acc, np.int64).reshape(-1, 3),
               sources=np.array(sources, float).reshape(-1, 2), keep_cp=keep_cp, cp_res=cp_res,
               ratio_pass=np.isin(np.arange(n_ep), list(passed)))
    if cfg.nhc_calib:
        out["mount_deg"] = np.degrees(fix.atVector(_graph.MOUNT))
    out["stats"] = dict(st, fix_rate=float(fix_ok.mean()), seconds=time.perf_counter() - t_all)
    log(f"FIX {100 * fix_ok.mean():.1f}% (rejected by residual {100 * fix_rej.mean():.1f}%), "
        f"total {out['stats']['seconds']:.1f}s")
    return out


__all__ = ["BatchConfig", "load", "load_imu_csv", "prepare", "run_batch", "save"]
