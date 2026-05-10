"""Robust M-estimator noise-model wrappers for GTSAM factors."""

from __future__ import annotations

import gtsam


def maybe_robust(base_noise, k, kind: str = 'huber'):
    """Wrap `base_noise` with an M-estimator; `k<=0` disables.

    Parameters
    ----------
    base_noise : gtsam.noiseModel.Base
        Underlying Gaussian noise (σ is baked in here).
    k : float
        Robust threshold in units of σ. 0 or negative disables wrapping.
    kind : {'huber', 'cauchy', 'dcs', 'tukey'}
        Huber — quadratic for |r/σ|<k, linear beyond (hard transition).
        Cauchy — weight 1/(1+(r/σ/k)²); smooth attenuation, GREAT-FGO style.
        DCS — Dynamic Covariance Scaling (adaptive σ per residual).
        Tukey — zero weight beyond k*σ.
    """
    if k <= 0:
        return base_noise
    mEst = gtsam.noiseModel.mEstimator
    kind = (kind or 'huber').lower()
    if kind == 'cauchy':
        est = mEst.Cauchy.Create(k)
    elif kind == 'dcs':
        est = mEst.DCS.Create(k)
    elif kind == 'tukey':
        est = mEst.Tukey.Create(k)
    else:
        est = mEst.Huber.Create(k)
    return gtsam.noiseModel.Robust.Create(est, base_noise)


def maybe_huber(base_noise, k):
    """Back-compat alias: always Huber. Prefer `maybe_robust` for new code."""
    return maybe_robust(base_noise, k, kind='huber')
