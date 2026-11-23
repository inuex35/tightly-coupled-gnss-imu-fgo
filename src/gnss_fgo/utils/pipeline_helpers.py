"""Small deterministic sort helpers shared across the gnss_fgo pipeline."""


def sorted_sys_ids(sig_map):
    """Deterministic system-id iteration order (GPS=1, GLO=2, ...)."""
    return sorted(sig_map.keys(), key=int)


def sorted_amb_items(amb_dict):
    """Deterministic ``(sat, freq) → key`` iteration sorted by sat then freq."""
    return sorted(amb_dict.items(),
                  key=lambda item: (int(item[0][0]), int(item[0][1])))


