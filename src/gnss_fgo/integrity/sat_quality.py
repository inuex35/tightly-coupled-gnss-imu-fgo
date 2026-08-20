"""Per-satellite observation-quality POLICIES.

One explicit class owns every sat-quality dict and the policies over
them (tick / observation-quality EWMA / CP-lock streaks). Constructed
with the runner's SatStateMap for future consolidation; note: an
attempted physical move of these dicts onto the map changed the
obsq EWMA trajectory with byte-identical inputs (unexplained — see
task notes) and was backed out.
"""



class SatQualityState:
    """Sat-quality / CP-hold policies over the runner's SatStateMap."""

    def __init__(self, states):
        self._states = states
        self.forced_hold_per_sat = set()
        self.cp_lock_streak = {}
        self.hold_quarantine = {}
        self.hold_streak_persat = {}
        self.persist_bad_streak = {}
        self.persist_bad_hold = {}
        self.obsq_ewma = {}
        self.obsq_bad_streak = {}
        self.recent_worst = {}
        self.recent_cppr = {}
        self.latest_el_deg = {}
        self.latest_snr_dbhz = {}


    def clear(self):
        self.cp_lock_streak.clear()
        self.hold_quarantine.clear()
        self.hold_streak_persat.clear()
        for d in (self.persist_bad_streak, self.persist_bad_hold,
                  self.obsq_ewma, self.obsq_bad_streak, self.recent_worst,
                  self.recent_cppr, self.latest_el_deg,
                  self.latest_snr_dbhz):
            d.clear()
        self.forced_hold_per_sat.clear()

    def tick(self, amb_keys_tc, info):
        """Advance hold/cooldown/probation maps and return forced-hold keys."""
        forced_hold = set()


        for key in list(self.hold_quarantine.keys()):
            rem = int(self.hold_quarantine.get(key, 0)) - 1
            if rem > 0:
                self.hold_quarantine[key] = rem
                forced_hold.add(key)
            else:
                self.hold_quarantine.pop(key, None)


        active_long_hold_sat = set()
        for s in list(self.persist_bad_hold.keys()):
            rem = int(self.persist_bad_hold.get(s, 0))
            if rem <= 0:
                self.persist_bad_hold.pop(s, None)
                continue
            active_long_hold_sat.add(int(s))
            rem -= 1
            if rem > 0:
                self.persist_bad_hold[s] = rem
            else:
                self.persist_bad_hold.pop(s, None)

        if active_long_hold_sat:
            for (s, f) in list(amb_keys_tc.keys()):
                if int(s) in active_long_hold_sat:
                    forced_hold.add((s, f))
            info['persistent_bad_sat_hold'] = len(active_long_hold_sat)

        self.forced_hold_per_sat = forced_hold
        return forced_hold

    def update_observation_quality(
            self, cfg, per_sat_res, worst_sat=None, cppr_sat=None,
            sat_el_deg=None, sat_snr_dbhz=None):
        """Update per-sat residual memory and short-term quality proxies."""
        alpha = float(getattr(cfg, 'obsq_ewma_alpha', 0.2))
        alpha = min(max(alpha, 0.0), 1.0)
        thr_obsq = float(getattr(cfg, 'obsq_bad_streak_thresh', 2.0))
        worst_decay = float(getattr(cfg, 'obsq_recent_worst_decay', 0.8))
        worst_decay = min(max(worst_decay, 0.0), 1.0)
        cppr_decay = float(getattr(cfg, 'obsq_recent_cppr_decay', 0.8))
        cppr_decay = min(max(cppr_decay, 0.0), 1.0)
        seen = set()
        for s, rmax in (per_sat_res or {}).items():
            s = int(s)
            seen.add(s)
            prev = float(self.obsq_ewma.get(s, 0.0))
            cur = float(rmax)
            self.obsq_ewma[s] = cur if prev <= 0 else ((1.0 - alpha) * prev + alpha * cur)
            if cur > thr_obsq:
                self.obsq_bad_streak[s] = int(self.obsq_bad_streak.get(s, 0)) + 1
            else:
                self.obsq_bad_streak[s] = 0
        for s in list(self.obsq_ewma.keys()):
            if s not in seen:
                self.obsq_ewma[s] = 0.0
        for s in list(self.obsq_bad_streak.keys()):
            if s not in seen:
                self.obsq_bad_streak[s] = 0
        active_sats = set(int(s) for s in (per_sat_res or {}).keys())
        if worst_sat is not None:
            active_sats.add(int(worst_sat))
        if cppr_sat:
            active_sats.update(int(s) for s in cppr_sat.keys())
        for s in active_sats:
            prev_w = float(self.recent_worst.get(s, 0.0) or 0.0)
            is_worst = 1.0 if worst_sat is not None and int(s) == int(worst_sat) else 0.0
            self.recent_worst[s] = (worst_decay * prev_w) + is_worst

            prev_c = float(self.recent_cppr.get(s, 0.0) or 0.0)
            cppr_cur = float((cppr_sat or {}).get(s, 0.0) or 0.0)
            self.recent_cppr[s] = (cppr_decay * prev_c) + cppr_cur
        for s in list(self.recent_worst.keys()):
            if s not in active_sats:
                self.recent_worst[s] = worst_decay * float(self.recent_worst.get(s, 0.0) or 0.0)
        for s in list(self.recent_cppr.keys()):
            if s not in active_sats:
                self.recent_cppr[s] = cppr_decay * float(self.recent_cppr.get(s, 0.0) or 0.0)
        for s, el_deg in (sat_el_deg or {}).items():
            self.latest_el_deg[int(s)] = float(el_deg)
        for s, snr_dbhz in (sat_snr_dbhz or {}).items():
            self.latest_snr_dbhz[int(s)] = float(snr_dbhz)



    def update_cp_lock(self, visible_keys, slip_keys=None, forced_hold=None):
        """Update per-(sat,freq) CP lock streak.

        This is independent of ambiguity-key lifetimes. A key keeps
        accumulating while the sat/freq stays visible and is not marked by
        a slip / hold path. Missing, slipped, or forcibly-held keys reset.
        """
        visible = set(visible_keys or ())
        slips = set(slip_keys or ())
        held = set(forced_hold or ())

        for key in list(self.cp_lock_streak.keys()):
            if key not in visible:
                self.cp_lock_streak.pop(key, None)

        for key in visible:
            if key in slips or key in held:
                self.cp_lock_streak[key] = 0
            else:
                self.cp_lock_streak[key] = int(self.cp_lock_streak.get(key, 0)) + 1

    def reset_cp_lock(self, keys):
        """Force-reset CP lock streak for the given sat/freq keys."""
        for key in set(keys or ()):
            self.cp_lock_streak[key] = 0



def get_sat_quality(tc):
    """Return the runner-owned sat-quality manager, creating it lazily."""
    sq = tc._sat_quality
    if sq is None:
        sq = SatQualityState(tc._sat_states)
        tc._sat_quality = sq
    return sq
