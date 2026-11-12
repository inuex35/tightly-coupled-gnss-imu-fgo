"""Batched integer least squares (LAMBDA) on cssrlib's mlambda.

Many independent problems (one per epoch) are spread over a process pool;
results are identical to calling mlambda on each problem in turn.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_MIN_PARALLEL = 256   # below this, pool start-up costs more than it saves


def _solve_one(a, Q):
    from cssrlib.core.mlambda import mlambda
    try:
        afix, s, _, _ = mlambda(np.asarray(a, float), np.asarray(Q, float), ncands=2)
    except (np.linalg.LinAlgError, ValueError):
        n = len(a)
        return np.zeros(n, int), np.zeros(n, int), np.zeros(2), -1
    return (np.rint(afix[:, 0]).astype(int), np.rint(afix[:, 1]).astype(int),
            np.asarray(s[:2], float), 0)


def _solve_chunk(chunk):
    return [_solve_one(a, Q) for a, Q in chunk]


def lambda_batch(ahats, Qs, workers=0):
    """Integer LS for each (ahat, Q).

    Returns (best, second, s, info): best/second integer vectors per problem,
    s (nprob, 2) squared residuals (ratio = s[:, 1] / s[:, 0]), info 0 = ok.
    workers: processes to use; 0 = os.cpu_count(), 1 = serial.
    """
    probs = list(zip(ahats, Qs))
    workers = workers or os.cpu_count() or 1
    if workers <= 1 or len(probs) < _MIN_PARALLEL:
        out = _solve_chunk(probs)
    else:
        size = -(-len(probs) // (4 * workers))
        chunks = [probs[i:i + size] for i in range(0, len(probs), size)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = [r for part in ex.map(_solve_chunk, chunks) for r in part]
    best = [o[0] for o in out]
    second = [o[1] for o in out]
    s = np.array([o[2] for o in out], dtype=float).reshape(-1, 2)
    info = np.array([o[3] for o in out], dtype=np.int32)
    return best, second, s, info
