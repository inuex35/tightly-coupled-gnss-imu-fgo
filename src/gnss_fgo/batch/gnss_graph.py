"""GNSS-only batch graph (positions + SD arcs); initial guess for the TC batch.

  X(k)  antenna ECEF position per epoch (Point3)
  N(a)  between-receiver SD float ambiguity per satellite/frequency arc [cyc]
  DoubleDifference{Pseudorange,CarrierPhase}Factor per DD row (Huber),
  BetweenFactorPoint3 random walk between consecutive epochs,
  weak priors on X(0) and on every SD ambiguity (DD gauge).
"""
import numpy as np
import gtsam
from gtsam.symbol_shorthand import N, X

from . import rows as R

DIMS = {"x": 3, "n": 1}


def huber_noise(sigma, k):
    return gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(k),
        gtsam.noiseModel.Isotropic.Sigma(1, sigma))


def add_ambiguities(graph, init, rows, sig_amb, seen=None):
    """N(a) per arc, seeded with SD carrier-minus-code, plus a weak prior."""
    seen = set() if seen is None else seen
    prior = gtsam.noiseModel.Isotropic.Sigma(1, sig_amb)
    for r in rows:
        lam = r[R.LAM]
        for a, cpr, cpb, prr, prb in (
                (int(r[R.ARC_REF]), r[R.CP_RR], r[R.CP_BR], r[R.PR_RR], r[R.PR_BR]),
                (int(r[R.ARC_TGT]), r[R.CP_RT], r[R.CP_BT], r[R.PR_RT], r[R.PR_BT])):
            if a not in seen:
                seen.add(a)
                sd = ((cpr - cpb) - (prr - prb)) / lam
                init.insert(N(a), float(sd))
                graph.add(gtsam.PriorFactorDouble(N(a), float(sd), prior))
    return seen


def build(rows, tow, rb, pos0, cfg):
    graph, init = gtsam.NonlinearFactorGraph(), gtsam.Values()
    base = gtsam.Point3(*rb)
    for k in range(len(tow)):
        init.insert(X(k), gtsam.Point3(*pos0))
        if k > 0:
            dt = max(tow[k] - tow[k - 1], 0.2)
            graph.add(gtsam.BetweenFactorPoint3(
                X(k - 1), X(k), gtsam.Point3(0, 0, 0),
                gtsam.noiseModel.Isotropic.Sigma(3, cfg.sig_rw * np.sqrt(dt / 0.2))))
    graph.add(gtsam.PriorFactorPoint3(
        X(0), gtsam.Point3(*pos0), gtsam.noiseModel.Isotropic.Sigma(3, 100.0)))
    add_ambiguities(graph, init, rows, cfg.sig_amb)
    for r in rows:
        k, lam, w = int(r[R.EPOCH]), r[R.LAM], r[R.WEIGHT]
        sr, st, sbr, sbt = R.sat_points(r)
        graph.add(gtsam.DoubleDifferencePseudorangeFactor(
            X(k), r[R.PR_RR], r[R.PR_BR], r[R.PR_RT], r[R.PR_BT], sr, st, sbr, sbt, base,
            huber_noise(cfg.sig_pr * w, cfg.huber)))
        graph.add(gtsam.DoubleDifferenceCarrierPhaseFactor(
            X(k), N(int(r[R.ARC_REF])), N(int(r[R.ARC_TGT])),
            r[R.CP_RR], r[R.CP_BR], r[R.CP_RT], r[R.CP_BT], sr, st, sbr, sbt, base, lam,
            huber_noise(cfg.sig_cp * w, cfg.huber)))
    return graph, init


def add_fix(graph, accepted, sig_fix):
    """Copy of graph + one BetweenFactorDouble N(t) - N(r) = n per relation."""
    fgraph = gtsam.NonlinearFactorGraph(graph)
    noise = gtsam.noiseModel.Isotropic.Sigma(1, sig_fix)
    for r, t, n in accepted:
        fgraph.add(gtsam.BetweenFactorDouble(N(r), N(t), float(n), noise))
    return fgraph


def positions(values, n_ep):
    return np.array([values.atPoint3(X(k)) for k in range(n_ep)])
