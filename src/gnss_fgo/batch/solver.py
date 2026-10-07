"""Levenberg-Marquardt over a whole-run graph: gtsam.cuda (cuDSS) or CPU."""
import time

import gtsam


def cuda_available():
    return hasattr(gtsam, "cuda")


def _lm_params(p, max_iters):
    p.setMaxIterations(max_iters)
    p.setRelativeErrorTol(1e-5)
    p.setAbsoluteErrorTol(1e-5)
    return p


def solve(graph, init, backend="auto", max_iters=200):
    """Optimize; returns (values, seconds, info string).

    backend: "cuda" (gtsam.cuda SparseLevenbergMarquardt with cuDSS; the
    graph is linearized on the CPU, so any factor works), "cpu" (METIS
    ordering + diagonal damping; the default settings stop early on these
    graphs) or "auto".
    """
    if backend == "auto":
        backend = "cuda" if cuda_available() else "cpu"
    t = time.perf_counter()
    if backend == "cuda":
        cu = gtsam.cuda
        p = _lm_params(cu.SparseLevenbergMarquardtParams(), max_iters)
        lin = cu.LinearSolverOptions()
        lin.backend = cu.LinearSolverType.Cudss
        p.linear = lin
        p.fallbackOnUnsupported = False
        opt = cu.SparseLevenbergMarquardtOptimizer(graph, init, p)
        # Copy: the returned Values keeps the optimizer (and its GPU buffers)
        # alive, which made later solves ~3x slower.
        v = gtsam.Values(opt.optimize())
        r = opt.result()
        info = (f"cuda iters={r.iterations} "
                f"term={str(r.termination).split('.')[-1]}")
    else:
        p = _lm_params(gtsam.LevenbergMarquardtParams(), max_iters)
        p.setOrderingType("METIS")
        p.setDiagonalDamping(True)
        opt = gtsam.LevenbergMarquardtOptimizer(graph, init, p)
        v = gtsam.Values(opt.optimize())
        info = f"cpu iters={opt.iterations()}"
    return v, time.perf_counter() - t, info
