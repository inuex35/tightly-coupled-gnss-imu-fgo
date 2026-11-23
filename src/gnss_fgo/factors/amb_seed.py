"""Clock-free fresh ambiguity seeding for persistent carrier-phase arcs."""


def seed_one_amb_prior(tc, graph, values, key_n, n0_seed, key_id, sigma_pr, lam):
    """Insert a fresh arc's ambiguity value + code-noise-derived prior (σ_PR / λ)."""
    sig = max(float(sigma_pr) / float(lam), 1.0)
    values.insert(key_n, n0_seed)
    graph.addPriorDouble(key_n, n0_seed, tc._noise1(sig))
    tc.current_epoch.seeded_amb_keys.add(key_id)


def init_dd_ambiguity_priors(tc, graph, values, amb_dict, new_amb,
                             existing_estimate, freq, lam, pair_sat_info,
                             sigma_pr=3.0):
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
        seed_one_amb_prior(tc, graph, values, key_n, n0_seed, key_id,
                           sigma_pr=sigma_pr, lam=lam)
        new_amb[key_id] = key_n
