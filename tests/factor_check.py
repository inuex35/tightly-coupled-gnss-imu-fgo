"""Generic numeric-vs-analytic Jacobian checker for ANY gtsam factor.

Usage:
    check_factor_jacobians(factor, values)          # assert all blocks match
    errs = jacobian_errors(factor, values)          # {key: max_abs_diff}

Works for factors over Pose3, Rot3, Point3/Vector, and scalar (Double)
variables — the perturbation is applied on the manifold (retract) where
one exists, additively otherwise. Whitening is undone via the factor's
noise model so the comparison happens on the raw error.

This exists because a hand-written NHC CustomFactor shipped a wrong
rotation Jacobian for months (R^T*skew(v) instead of skew(R^T v)) and
nothing caught it: ISAM2 happily descends along a wrong gradient. Any
new hand-written factor should get a one-line test through this helper.
"""
import numpy as np
import gtsam


def _get(values, key):
    for getter in (values.atPose3, values.atRot3, values.atPoint3,
                   values.atVector, values.atDouble):
        try:
            return getter(key)
        except Exception:
            continue
    raise TypeError(f"unsupported variable type for key {key}")


def _dim(val):
    if isinstance(val, gtsam.Pose3):
        return 6
    if isinstance(val, gtsam.Rot3):
        return 3
    if isinstance(val, float):
        return 1
    return np.asarray(val).size


def _perturb(val, delta):
    if isinstance(val, (gtsam.Pose3, gtsam.Rot3)):
        return val.retract(delta)
    if isinstance(val, float):
        return val + float(delta[0])
    return np.asarray(val, dtype=float) + delta


def _replace(values, key, val):
    out = gtsam.Values(values)
    out.erase(key)
    if isinstance(val, float):
        out.insert(key, float(val))
    else:
        out.insert(key, val)
    return out


def _unwhiten_sigmas(factor):
    model = factor.noiseModel()
    robust_base = getattr(model, 'noise', None)
    if callable(robust_base):        # Robust wrapper: use the base model
        model = model.noise()
    return np.asarray(model.sigmas(), dtype=float)


def jacobian_errors(factor, values, eps=1e-5):
    """Max |analytic - numeric| per variable, on the unwhitened error."""
    sig = _unwhiten_sigmas(factor)
    A, _ = factor.linearize(values).jacobian()
    keys = list(factor.keys())
    errs = {}
    col = 0
    for key in keys:
        val = _get(values, key)
        d = _dim(val)
        J_ana = A[:, col:col + d] * sig[:, None]
        J_num = np.zeros_like(J_ana)
        for i in range(d):
            step = np.zeros(d); step[i] = eps
            vp = _replace(values, key, _perturb(val, step))
            vm = _replace(values, key, _perturb(val, -step))
            J_num[:, i] = (factor.unwhitenedError(vp)
                           - factor.unwhitenedError(vm)) / (2 * eps)
        errs[key] = float(np.abs(J_ana - J_num).max())
        col += d
    return errs


def check_factor_jacobians(factor, values, atol=1e-4, eps=1e-5):
    """Assert every Jacobian block matches its numeric differentiation."""
    errs = jacobian_errors(factor, values, eps=eps)
    bad = {k: e for k, e in errs.items() if e > atol}
    assert not bad, f"Jacobian mismatch (atol={atol}): {bad}"
    return errs
