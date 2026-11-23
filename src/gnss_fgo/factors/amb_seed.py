"""Clock-free fresh ambiguity seeding for persistent carrier-phase arcs.

An ambiguity variable N(sat, freq, arc_gen) is created once when a continuous
arc starts (or after a cycle slip / outage expiry) and reused directly by all
subsequent epochs of that arc — no per-epoch key rotation, no continuing prior,
and no BetweenFactor chain.
"""


def seed_one_amb_prior(tc, graph, values, key_n, n0_seed, key_id):
    """Insert a fresh arc's ambiguity value + loose clock-free prior."""
    sig = tc.cfg.sigma_amb0 if tc.phase == 2 else 3.0
    values.insert(key_n, n0_seed)
    graph.addPriorDouble(key_n, n0_seed, tc._noise1(sig))
    tc.current_epoch.seeded_amb_keys.add(key_id)


def init_dd_ambiguity_priors(tc, graph, values, amb_dict, new_amb,
                             existing_estimate, freq, lam, pair_sat_info):
    """Ensure both satellites of a DDCP pair have live ambiguity keys.

    If a key already exists in ``existing_estimate`` (the smoother already
    holds this arc's variable), reuse it untouched. Otherwise seed it once
    with the clock-free ``(SD_phase - SD_code) / lam`` estimate.
    """
    for sat_id, key_n, cp_rover, cp_base, pr_rover, pr_base in pair_sat_info:
        key_id = (sat_id, freq)
        if key_id in new_amb or values.exists(key_n):
            continue
        if existing_estimate is not None and existing_estimate.exists(key_n):
            new_amb[key_id] = key_n
            continue
        if key_id in amb_dict and amb_dict[key_id] == key_n and existing_estimate is None:
            # Phase 1 / init batch: already seeded earlier in this graph.
            continue
        n0_seed = ((cp_rover - cp_base) - (pr_rover - pr_base)) / lam
        seed_one_amb_prior(tc, graph, values, key_n, n0_seed, key_id)
        new_amb[key_id] = key_n
