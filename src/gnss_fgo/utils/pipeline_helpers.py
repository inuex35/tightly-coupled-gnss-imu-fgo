"""Small shared helpers across the gnss_fgo sequential and batch pipelines."""

import gtsam


def huber_noise(sigma: float, k: float):
    """Build 1-D Gaussian or GTSAM Huber robust noise model."""
    base = gtsam.noiseModel.Isotropic.Sigma(1, float(sigma))
    if float(k) <= 0.0:
        return base
    return gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(float(k)), base)


def sorted_sys_ids(sig_map):
    """Deterministic system-id iteration order (GPS=1, GLO=2, ...)."""
    return sorted(sig_map.keys(), key=int)


def sorted_amb_items(amb_dict):
    """Deterministic ``(sat, freq) → key`` iteration sorted by sat then freq."""
    return sorted(amb_dict.items(),
                  key=lambda item: (int(item[0][0]), int(item[0][1])))


