"""Small helpers shared across the gnss_fgo pipeline.

The deterministic sort helpers exist so every consumer iterates
satellites and ambiguities in one identical order — reproducibility
work upstream relies on that, so all callers must share these
byte-for-byte rather than keep private copies.
"""

import numpy as np


def sorted_sys_ids(sig_map):
    """Deterministic system-id iteration order (GPS=1, GLO=2, ...)."""
    return sorted(sig_map.keys(), key=int)


def sorted_amb_items(amb_dict):
    """Deterministic ``(sat, freq) → key`` iteration sorted by sat then freq."""
    return sorted(amb_dict.items(),
                  key=lambda item: (int(item[0][0]), int(item[0][1])))


def heading_from_pose(pose):
    """Body forward-axis heading [deg] from a Pose3 body→ENU rotation."""
    r_body2enu = np.array(pose.rotation().matrix())
    return float(np.degrees(np.arctan2(r_body2enu[0, 0], r_body2enu[1, 0])))


