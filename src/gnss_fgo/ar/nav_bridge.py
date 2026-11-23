"""Publish live smoother ambiguity estimates to cssrlib's nav state for AR."""

from ..utils import sorted_amb_items


def publish_attempt(tc, sat_list, result):
    """Record ratio stats and cssrlib ddidx flags after a LAMBDA attempt."""
    tc.ddidx(tc.nav, sat_list)
    tc._last_s0, tc._last_s1 = result.s0, result.s1


def publish_marginals(tc, estimate, key_pose, amb_dict):
    """Populate nav.x and nav.vsat from live persistent smoother ambiguities."""
    tc._ar_key_pose = key_pose
    tc.nav.P[:, :] = 0
    tc.nav.vsat[:, :] = 0
    tc.nav.x[tc.nav.na:] = 0
    cp_visible = tc._ar_cp_visible_sf
    for (s, f), k in sorted_amb_items(amb_dict):
        if estimate.exists(k):
            tc.nav.x[tc.IB(s, f, tc.nav.na)] = estimate.atDouble(k)
            if ((s, f) not in tc.current_epoch.seeded_amb_keys
                    and (not cp_visible or (int(s), int(f)) in cp_visible)):
                tc.nav.vsat[s - 1, f] = 1
