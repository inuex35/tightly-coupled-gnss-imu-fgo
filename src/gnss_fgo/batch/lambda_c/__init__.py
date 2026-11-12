"""Batched integer least squares: RTKLIB lambda.c (C, OpenMP) with a Python fallback.

lambda.c is RTKLIB 2.4.3 (BSD 2-clause, see LICENSE_RTKLIB.txt); rtklib.h is a
minimal stand-in for the matrix helpers it needs, and lambda_batch.c solves
many problems in one call. Build once (or let `build()` do it):

    gcc -O3 -march=native -fPIC -shared -fopenmp -o liblambda.so lambda_batch.c -lm

Without the shared library the cssrlib mlambda (pure Python, ~250x slower,
same integers) is used.
"""
import ctypes
import os
import subprocess

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_DIR, "liblambda.so")
_lib = None


def build(cc="gcc"):
    """Compile liblambda.so next to the sources. Returns True on success."""
    cmd = [cc, "-O3", "-fPIC", "-shared", "-fopenmp", "-o", _SO,
           os.path.join(_DIR, "lambda_batch.c"), "-lm"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _load():
    global _lib
    if _lib is None and os.path.exists(_SO):
        lib = ctypes.CDLL(_SO)
        dp = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        ip = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        lp = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
        lib.lambda_batch.argtypes = [ctypes.c_int, ip, lp, lp, dp, dp, dp, dp, ip]
        lib.lambda_batch.restype = None
        _lib = lib
    return _lib


def available():
    return _load() is not None


def lambda_batch(ahats, Qs, backend="auto"):
    """Integer LS for each (ahat, Q).

    Returns (best, second, s, info): best/second integer vectors per problem,
    s (nprob, 2) squared residuals (ratio = s[:, 1] / s[:, 0]), info 0 = ok.
    backend: "c", "python" or "auto" (C when liblambda.so is present).
    """
    if backend == "python" or (backend == "auto" and not available()):
        return _lambda_python(ahats, Qs)
    lib = _load()
    if lib is None:
        raise RuntimeError("liblambda.so not built; run gnss_fgo.batch.lambda_c.build()")
    n = np.array([len(a) for a in ahats], dtype=np.int32)
    off_a = np.zeros(len(n), dtype=np.int64)
    off_q = np.zeros(len(n), dtype=np.int64)
    off_a[1:] = np.cumsum(n[:-1])
    off_q[1:] = np.cumsum(n[:-1].astype(np.int64) ** 2)
    a = np.ascontiguousarray(np.concatenate(ahats), dtype=np.float64)
    Q = np.ascontiguousarray(np.concatenate(
        [np.asarray(q, float).ravel(order="F") for q in Qs]))
    F = np.zeros(2 * len(a))
    s = np.zeros(2 * len(n))
    info = np.zeros(len(n), dtype=np.int32)
    lib.lambda_batch(len(n), n, off_a, off_q, a, Q, F, s, info)
    best, second = [], []
    for i, m in enumerate(n):
        f = F[2 * off_a[i]: 2 * off_a[i] + 2 * m].reshape(2, m)   # column-major n x 2
        best.append(np.rint(f[0]).astype(int))
        second.append(np.rint(f[1]).astype(int))
    return best, second, s.reshape(-1, 2), info


def _lambda_python(ahats, Qs):
    from cssrlib.core.mlambda import mlambda
    best, second, s_all, info = [], [], [], []
    for a, q in zip(ahats, Qs):
        afix, s, _, _ = mlambda(np.asarray(a, float), np.asarray(q, float), ncands=2)
        best.append(np.rint(afix[:, 0]).astype(int))
        second.append(np.rint(afix[:, 1]).astype(int))
        s_all.append(s[:2])
        info.append(0)
    return best, second, np.array(s_all, dtype=float).reshape(-1, 2), np.array(info, np.int32)
