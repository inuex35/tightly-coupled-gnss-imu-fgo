"""Float-ambiguity seeding for the DD graph — values, priors and holds.

Contract
--------
``init_dd_ambiguity_priors`` is called once per (ref, j, freq) pair before
the DD-CP factor is emitted. For each of the two satellites it decides
whether the epoch's N key needs a value + prior, and which of three seed
modes applies:

1. **continuing** — the (sat, freq) has a value in ``prev_amb_values``
   (previous epoch's estimate): reuse it with the tight ``sigma_cont``
   prior. This is the BetweenN-chain steady state.
2. **release re-seed** — a fix-and-hold was just released
   (``release_seed_pending``): one-shot prior at the last held integer,
   σ=0.1 cyc.
3. **fresh** — first sight (or post-reset): seed from the observations
   with the loose ``sigma_amb0`` prior (3 cyc in Phase 1).

The fresh seed is ``(SD_phase − SD_code) / λ`` — cssrlib-udbias style.
Both terms carry the same receiver SD clock, so the seed LEVEL is
clock-free. Never seed against geometric range: that bakes c·(dtr−dtb)
of the seed epoch into the N level, and with a drifting receiver clock
the level gauge of a fix-and-hold era diverges km-scale from a later
re-seed cohort — every held×free DD-CP pair then injects the gap into
the pose (run1 ep1619: 6.1 km gap, single-epoch 2.4 km jump).

Held satellites are skipped (their integer is pinned outside the graph).

Reads:  tc.cfg (sigma_cont / sigma_amb0), tc.phase,
        tc.epoch, per-sat ``SatState`` hold fields.
Writes: ``values`` / ``graph`` (N inserts + priors), ``new_amb``,
        ``SatState.amb_init_epoch / release_seed_pending``.
"""


def seed_one_amb_prior(tc, graph, values, sat_st, key_n, n0_seed,
                       prev_amb_values, key_id):
    """Insert one N value + prior using the three-mode policy above."""
    if prev_amb_values is not None and key_id in prev_amb_values:
        n0 = prev_amb_values[key_id][1]
        values.insert(key_n, n0)
        graph.addPriorDouble(key_n, n0, tc._noise1(tc.cfg.sigma_cont))
        return
    if (sat_st.release_seed_pending
            and sat_st.last_held_value is not None):
        n0 = sat_st.last_held_value
        values.insert(key_n, n0)
        graph.addPriorDouble(key_n, n0, tc._noise1(0.1))
        sat_st.amb_init_epoch = tc.epoch
        sat_st.release_seed_pending = False
        return
    sig = tc.cfg.sigma_amb0 if tc.phase == 2 else 3.0
    values.insert(key_n, n0_seed)
    graph.addPriorDouble(key_n, n0_seed, tc._noise1(sig))
    sat_st.amb_init_epoch = tc.epoch


def init_dd_ambiguity_priors(tc, graph, values, amb_dict, new_amb,
                             prev_amb_values, freq, lam, pair_sat_info):
    """Seed N values/priors for one pair's two satellites (see module doc)."""
    for sat_id, key_n, cp_rover, cp_base, pr_rover, pr_base in pair_sat_info:
        key_id = (sat_id, freq)
        sat_st = tc._sat_states.get(*key_id)
        # Clock-free level: SD phase minus SD code (NOT geometry).
        n0_seed = ((cp_rover - cp_base) - (pr_rover - pr_base)) / lam
        if sat_st.held_value is not None:
            continue
        if key_id in amb_dict or key_id in new_amb:
            continue
        if values.exists(key_n):
            continue
        seed_one_amb_prior(tc, graph, values, sat_st, key_n, n0_seed,
                           prev_amb_values, key_id)
        new_amb[key_id] = key_n
