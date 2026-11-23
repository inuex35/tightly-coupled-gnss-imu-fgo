"""Clock-free fresh ambiguity seeding for persistent carrier-phase arcs."""

_SIGMA_AMB0_P2 = 30.0  # loose gauge prior [cyc] for a fresh Phase-2 arc
_SIGMA_AMB0_P1 = 3.0   # Phase-1 stationary bootstrap prior [cyc]


def seed_one_amb_prior(tc, graph, values, key_n, n0_seed, key_id):
    """Insert a fresh arc's ambiguity value + loose clock-free prior."""
    sig = _SIGMA_AMB0_P2 if tc.phase == 2 else _SIGMA_AMB0_P1
    values.insert(key_n, n0_seed)
    graph.addPriorDouble(key_n, n0_seed, tc._noise1(sig))
    tc.current_epoch.seeded_amb_keys.add(key_id)


def init_dd_ambiguity_priors(tc, graph, values, amb_dict, new_amb,
                             existing_estimate, freq, lam, pair_sat_info):
    """Reuse existing smoother arc keys untouched; seed new arcs once clock-free."""
    for sat_id, key_n, cp_rover, cp_base, pr_rover, pr_base in pair_sat_info:
        key_id = (sat_id, freq)
        if key_id in new_amb or values.exists(key_n):
            continue
        if existing_estimate is not None and existing_estimate.exists(key_n):
            new_amb[key_id] = key_n
            continue
        if key_id in amb_dict and amb_dict[key_id] == key_n and existing_estimate is None:
            continue
        n0_seed = ((cp_rover - cp_base) - (pr_rover - pr_base)) / lam
        seed_one_amb_prior(tc, graph, values, key_n, n0_seed, key_id)
        new_amb[key_id] = key_n
