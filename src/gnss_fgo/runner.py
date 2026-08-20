"""ImuGnssTc — two-phase IMU/GNSS tight coupling runner."""

import numpy as np
import gtsam
from cssrlib.rtk import rtkpos
from cssrlib.gnss import sat2prn, ecef2pos, timediff

from .utils import (
    collect_imu_samples as _utils_collect_imu_samples,
    build_pim_from_samples as _utils_build_pim_from_samples,
    estimate_stationary_bias as _utils_estimate_stationary_bias,
    compute_gdop as _utils_compute_gdop,
    DDPRContext as _DDPRContext,
    ddpr_only_position as _utils_ddpr_only_position,
    heading_from_vel as _utils_heading_from_vel,
    make_imu_params as _utils_make_imu_params,
)
from .config import TcConfig
from .factors.epoch_context import make_epoch_diagnostics, prepare_process_epoch
from .factors import factors as _tc_factors
from . import initialization as _initialization
from . import pipeline as _pipeline
from .integrity import recovery as _tc_recovery
from .state.runtime_state import (
    MresSignalsState, RecoveryState, SatFieldView, SatStateMap,
)
from .integrity.sat_quality import SatQualityState
from .pipeline import update_smoother as _tc_isam
from .factors import prefit as _tc_prefit
from .utils import sorted_sys_ids


class ImuGnssTc:
    """Two-phase IMU/GNSS tight coupling processor.

    The estimator is the factor graph; cssrlib is a library it calls, not a
    base class it is. The complete surface this pipeline uses from cssrlib's
    engine is the delegation block below -- ten methods and the ratio stash
    -- everything else (EKF time/measurement updates, the engine's own
    process loop) is deliberately out of reach.
    """

    # Symbol helpers
    Xp = staticmethod(lambda i: gtsam.symbol('x', i))   # Phase 1 pose
    Xpose = staticmethod(lambda i: gtsam.symbol('P', i)) # Phase 2 pose
    Vel = staticmethod(lambda i: gtsam.symbol('v', i))
    Bias = staticmethod(lambda i: gtsam.symbol('b', i))
    N = staticmethod(lambda s, f, gen=0: gtsam.symbol(
        'n', int(gen) * 1000000 + int(s) * 10 + int(f)))
    Clk = staticmethod(lambda i: gtsam.symbol('c', i))  # rcv clock bias [s]

    # ── The cssrlib boundary ────────────────────────────────────────
    # Every capability this pipeline takes from the engine, in one block.
    # Measurement preparation and validation:
    def prepare_double_difference_measurements(self, *a, **k):
        return self.engine.prepare_double_difference_measurements(*a, **k)

    def zdres(self, *a, **k):
        return self.engine.zdres(*a, **k)

    def sdres(self, *a, **k):
        return self.engine.sdres(*a, **k)

    def valpos(self, *a, **k):
        return self.engine.valpos(*a, **k)

    # Ambiguity resolution (the cssrlib path; ar/ is the native one):
    def resamb_lambda(self, *a, **k):
        return self.engine.resamb_lambda(*a, **k)

    def resamb_lambda_rtklib(self, *a, **k):
        return self.engine.resamb_lambda_rtklib(*a, **k)

    def ddidx(self, *a, **k):
        return self.engine.ddidx(*a, **k)

    def holdamb_flags(self, *a, **k):
        return self.engine.holdamb_flags(*a, **k)

    def IB(self, *a, **k):
        return self.engine.IB(*a, **k)

    # The ratio stash lives on the engine (resamb_lambda writes it there);
    # forwarding keeps one source of truth for both AR paths.
    @property
    def _last_s0(self):
        return self.engine._last_s0

    @_last_s0.setter
    def _last_s0(self, v):
        self.engine._last_s0 = v

    @property
    def _last_s1(self):
        return self.engine._last_s1

    @_last_s1.setter
    def _last_s1(self, v):
        self.engine._last_s1 = v

    _NOISE1_CACHE = {}

    @staticmethod
    def _noise1(sigma):
        sigma_f = float(sigma)
        nm = ImuGnssTc._NOISE1_CACHE.get(sigma_f)
        if nm is None:
            nm = gtsam.noiseModel.Isotropic.Sigma(1, sigma_f)
            ImuGnssTc._NOISE1_CACHE[sigma_f] = nm
        return nm

    def __init__(self, nav, pos0, base_ecef, imu_data,
                 lever_arm=np.zeros(3), logfile=None, cfg=None):
        """Initialize TC processor."""
        self.engine = rtkpos(nav, pos0, logfile)
        self.nav = self.engine.nav

        self.base_ecef = np.array(base_ecef)
        self.imu_data = imu_data
        self.lever_arm = np.array(lever_arm, dtype=float)

        self._init_base_frame()

        # Config: use provided TcConfig or load from env vars
        self.cfg = cfg if cfg is not None else TcConfig.from_env()
        self._apply_nav_config()
        self.body_rot_std = self.cfg.body_rot_std
        self.accel_noise = self.cfg.accel_noise
        self.gyro_noise = self.cfg.gyro_noise
        self.accel_bias_sigma = self.cfg.accel_bias_sigma
        self.gyro_bias_sigma = self.cfg.gyro_bias_sigma
        self._recovery = RecoveryState()
        self.total_factor_count = 0
        self._mres_signals = MresSignalsState()

        self._init_runtime_state()

    def _init_base_frame(self):
        """Build the ENU frame anchored at the configured base station."""
        lat, lon, _ = ecef2pos(self.base_ecef)
        sl, cl = np.sin(lat), np.cos(lat)
        sn, cn = np.sin(lon), np.cos(lon)
        self.R_enu2ecef = np.array([
            [-sn, -sl * cn, cl * cn],
            [cn, -sl * sn, cl * sn],
            [0, cl, sl]])
        self.ecef_T_nav = gtsam.Pose3(
            gtsam.Rot3(self.R_enu2ecef), gtsam.Point3(*self.base_ecef))
        self.base_pt = gtsam.Point3(*self.base_ecef)

    def _apply_nav_config(self):
        """Apply TcConfig values onto cssrlib's mutable nav state."""
        self.nav.armode = self.cfg.ar_mode
        self.nav.parmode = self.cfg.parmode
        self.nav.elmin = np.deg2rad(float(self.cfg.elmin_deg))
        self.nav.cnr_min = float(self.cfg.cnr_min_dbhz)
        self.nav.par_P0 = self.cfg.par_P0
        self.nav.thresar = self.cfg.ar_thresar
        self.nav.rtklib_mode = bool(self.cfg.rtklib_mode)
        self.nav.arfilter = bool(self.cfg.ar_arfilter)
        self.nav.minfixsats = self.cfg.ar_minfixsats
        self.nav.valpos_thres = float(self.cfg.valpos_thres)

    def _init_runtime_state(self):
        """Initialize mutable per-run state for both pipeline phases."""
        # State
        self.phase = 1
        self.epoch = 0
        self.imu_idx = 0
        self.tc_epoch = 0

        self.phase1_t = 0.0
        self.isam = self._make_isam2(self.cfg.phase1_fls_lag,
                                       self.cfg.isam2_relinearize_skip,
                                       self.cfg.isam2_relinearize_threshold)
        self.amb_keys = {}
        self._isam_p1_inserted = set()

        # Doppler clock-bias chain: last epoch that owns a Clk key, and the
        # keys this epoch's factors reach back to (re-stamped by the FLS).
        self._doppler_clk_last = None
        self._doppler_keep_keys = []
        self._doppler_cb_prev = None

        # Collecting state (between Phase 1 and Phase 2)
        self.collecting = False
        self.collected_fixes = []  # list of {'ecef': ..., 'vel': ..., 'imu': [...]}
        self.collect_imu_buf = []  # IMU samples for current collection epoch

        self._sat_states = SatStateMap()
        self._amb_key_view = SatFieldView(self._sat_states, 'amb_key')
        self._amb_gen_view = SatFieldView(self._sat_states, 'amb_gen', absent=0)
        self._amb_lam_view = SatFieldView(self._sat_states, 'amb_lam', absent=0.0)
        self._amb_init_epoch_view = SatFieldView(
            self._sat_states, 'amb_init_epoch')
        self._rejc_cp_pr_view = SatFieldView(
            self._sat_states, 'rejc_cp_pr', absent=0)
        self._fix_streak_view = SatFieldView(
            self._sat_states, 'fix_streak', absent=0)

        # Phase 2 (initialized at transition)
        self.isam2 = None
        self.imu_params = None
        self.tc_bias = None
        self.lever_arm_tc = gtsam.Point3(0, 0, 0)

        self._last_obs_t = None  # for real-seconds dt tracking
        self._epoch_dt = 0.2     # actual seconds since last process() call
        # Per-epoch scratch (see _reset_epoch_scratch): the historical
        # every-epoch wipe doubled as the initializer, so with a
        # persist_* flag on these must exist before the first epoch.
        self.ref_sats = {}
        self.amb_gen = {}
        self.amb_lam = {}
        self.amb_init_epoch = {}
        self._sat_quality = None

        self._init_epoch_state_defaults()

    def _init_epoch_state_defaults(self):
        """Per-epoch / per-run state previously read via ``getattr(tc, '_X', default)`` shims throughout the package."""
        # AR diagnostics (set inside ar.run_ar)
        self._ar_context_reject = None
        self._ar_subset_debug = None
        self._last_ar_outcome = 'not_called'
        # Per-epoch DDCP / DDPR factor bookkeeping (reset in build_dd_factors)
        self._ar_cp_visible_sf = set()
        self._last_custom_ddcp_local = set()
        self._last_custom_ddcp_global = {}
        self._last_cp_pr_reject = 0
        self._last_hold_gauge_rel = []
        self._last_rejc_wipe = 0
        self._last_ddpr_sat_tags = []
        self._last_main_ddpr_res = 0.0
        self._last_main_ddpr_per_sat = {}
        self._last_main_ddpr_epoch = -10**9
        self._last_per_sat_res = {}
        self._cached_ddpr_res_pre = None
        # write_marginals diagnostics (cp visibility hysteresis)
        self._cp_visible_sf_last_ep = {}
        # FLS-update timing accumulators
        self._fls_update_calls = 0
        self._fls_update_time_total = 0.0
        self._fls_update_last_ms = 0.0
        # Phase-2 init bootstrap counters (filled by transition_to_tc)
        self._tc_bootstrap_ddpr_epochs = 0
        self._tc_fresh_amb_epochs = 0
        # Phase-1 last solution (used by velocity estimator)
        self._last_sol_ecef = None
        # Reset-flag for per-Phase-2 amb cleanup (build_dd_factors first call)
        self._rejc_reset_at_p2 = False
        # GF cycle-slip detector running state {sat: [last_gf, n, mean_gf]}
        self._gf_state = {}
        self._last_s0 = 0.0
        self._last_s1 = 0.0

    def _update_epoch_dt(self, obs):
        """Refresh self._epoch_dt with elapsed seconds since the previous
        obs epoch. dt only — the per-epoch state wipes live in
        _reset_epoch_scratch (review finding A-1)."""
        if self._last_obs_t is not None:
            try:
                dt = float(timediff(obs.t, self._last_obs_t))
                if dt > 0:
                    self._epoch_dt = dt
            except (TypeError, ValueError):
                pass
        self._last_obs_t = obs.t

    def _reset_epoch_scratch(self):
        """Historical every-epoch wipes, now individually gated.

        For most of this project's life ALL of these were wiped every
        epoch, silently disabling ref-sat continuity, ar_wait_new and
        the sat-quality subsystem (review finding A-1). The persist_*
        flags default to the historical wipe so the published numbers
        stand; flip individually for a measured A/B.
        total_factor_count is never reset here — zeroing the cumulative
        counter once made the held-CP FDE bookkeeping silently inert.
        """
        cfg = self.cfg
        self.thresslip = cfg.thres_slip
        self.cmc_thresh = cfg.cmc_thresh
        self.cn0_min = cfg.cn0_min
        self.ar_wait_new = cfg.ar_wait_new
        if not cfg.persist_ref_sats:
            self.ref_sats = {}
        if not cfg.persist_amb_gen:
            self.amb_gen = {}
        self.amb_lam = {}
        self.amb_init_epoch = {}
        if not cfg.persist_sat_quality or self._sat_quality is None:
            self._sat_quality = SatQualityState(self._sat_states)


    def _assign_view(self, view: SatFieldView, value):
        if value is view:
            return
        view.clear()
        if value:
            for k, v in value.items():
                view[k] = v

    # ── Generated forwarders ───────────────────────────────────────
    # Per-(sat,freq) dict views onto SatStateMap and scalar fields of the
    # grouped state dataclasses, exposed as flat tc attributes. One table
    # instead of 34 hand-written property/setter pairs; semantics are
    # identical (view assignment goes through _assign_view).
    _VIEW_FORWARDS = {
        'amb_keys_tc': '_amb_key_view',
        'amb_gen': '_amb_gen_view',
        'amb_lam': '_amb_lam_view',
        'amb_init_epoch': '_amb_init_epoch_view',
        'rejc_cp_pr': '_rejc_cp_pr_view',
        '_fix_streak': '_fix_streak_view',
    }
    _FIELD_FORWARDS = {
        'skip_count': ('_recovery', 'skip_count'),
        '_recov_cp_hold': ('_recovery', 'recov_cp_hold'),
        '_recov_cp_release_streak': ('_recovery', 'recov_cp_release_streak'),
        '_pim_discontinuity': ('_recovery', 'pim_discontinuity'),
        '_ddpr_bad_count': ('_recovery', 'ddpr_bad_count'),
        '_last_main_ddpr_res': ('_mres_signals', 'last_res'),
        '_last_main_ddpr_per_sat': ('_mres_signals', 'per_sat'),
        '_last_main_ddpr_epoch': ('_mres_signals', 'epoch'),
    }



    def _make_imu_params(self):
        """Thin adapter — see utils.imu.make_imu_params."""
        return _utils_make_imu_params(
            accel_noise=self.accel_noise,
            gyro_noise=self.gyro_noise,
            accel_bias_sigma=self.accel_bias_sigma,
            gyro_bias_sigma=self.gyro_bias_sigma,
            scale=self.cfg.imu_scale,
            integ_cov=self.cfg.imu_integ_cov)

    def _antenna_ecef(self, pose, ecef_body):
        """ECEF antenna position = body ECEF + R_body @ lever. Lever=0 → passthrough."""
        lever_arr = np.array(self.lever_arm_tc) \
            if getattr(self, 'lever_arm_tc', None) is not None \
            else np.zeros(3)
        if np.linalg.norm(lever_arr) == 0:
            return ecef_body
        R_body = self.ecef_T_nav.compose(pose).rotation().matrix()
        return ecef_body + R_body @ lever_arr

    def _compute_max_dd_frac(self, estimate, obs_sd, sat, ns):
        """Max |DD_N − round(DD_N)| across systems+freqs — indicator of pose bias."""
        max_frac = 0.0
        for sys_id in sorted_sys_ids(obs_sd.sig):
            idx_sys = [i for i in range(ns) if sat2prn(sat[i])[0] == sys_id]
            ref_s = self.ref_sats.get(sys_id)
            if ref_s is None or len(idx_sys) < 2:
                continue
            lams = _tc_factors.get_wavelengths(self, obs_sd, ref_s)
            for f in range(self.nav.nf):
                if f >= len(lams):
                    continue
                k_ref = self._sat_states.at(ref_s, f).amb_key
                if k_ref is None or not estimate.exists(k_ref):
                    continue
                n_ref = estimate.atDouble(k_ref)
                for ji in idx_sys:
                    js = sat[ji]
                    if js == ref_s:
                        continue
                    k_j = self._sat_states.at(js, f).amb_key
                    if k_j is None or not estimate.exists(k_j):
                        continue
                    dd_cyc = n_ref - estimate.atDouble(k_j)
                    frac = abs(dd_cyc - round(dd_cyc))
                    if frac > max_frac:
                        max_frac = frac
        return max_frac

    def _collect_imu_samples(self, n_samples=100, target_tow=None):
        """Thin adapter — see utils.imu.collect_imu_samples. Advances imu_idx."""
        samples, self.imu_idx = _utils_collect_imu_samples(
            self.imu_data, self.imu_idx,
            n_samples=n_samples, target_tow=target_tow)
        return samples

    def _estimate_stationary_bias(self):
        """Thin adapter — see utils.imu.estimate_stationary_bias."""
        return _utils_estimate_stationary_bias(
            self.imu_data, n_max=min(3000, self.imu_idx))

    def _build_pim_from_samples(self, bias, imu_samples):
        """Thin adapter — see utils.imu.build_pim_from_samples."""
        return _utils_build_pim_from_samples(self.imu_params, bias, imu_samples)

    def _heading_from_vel(self, vel, fallback, disp_enu=None):
        return _utils_heading_from_vel(vel, fallback, disp_enu)

    def _ddpr_only_position(self, obs, obsb, obs_sd, rs, rsb,
                              sat, el, iu, ir_map, pose_init):
        """Standalone DDPR-only LS. See utils.ls_solvers.ddpr_only_position."""
        lever = (self.lever_arm_tc
                 if getattr(self, 'lever_arm_tc', None) is not None
                 else gtsam.Point3(0, 0, 0))
        ctx = _DDPRContext(
            R_enu2ecef=self.R_enu2ecef,
            base_ecef=self.base_ecef,
            base_pt=self.base_pt,
            ecef_T_nav=self.ecef_T_nav,
            lever=lever,
            nav_nf=self.nav.nf,
            sigma_pr=self.cfg.sigma_pr,
            huber_pr=self.cfg.huber_pr,
            pr_robust_kind=self.cfg.pr_robust_kind,
            fde_pr=self.cfg.fde_pr,
            pick_ref_sat_idx=lambda *a, **k: _tc_prefit.pick_ref_sat_idx(self, *a, **k),
        )
        ecef_out, n_active, res_rms, _, _ = _utils_ddpr_only_position(
            obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
            pose_init, ctx)
        return ecef_out, n_active, res_rms

    def _compute_gdop(self, pred, ns, rs, iu, R_enu2ecef):
        """GDOP at IMU-predicted pose. See utils.geometry.compute_gdop."""
        return _utils_compute_gdop(
            np.array(pred.pose().translation()), ns, rs, iu,
            R_enu2ecef, self.base_ecef)

    def process(self, obs, obsb, rs, vs, dts, rsb, sat, el, iu, obs_sd,
                ir_map, ref_ecef=None):
        """Process one epoch. Returns (sol_ecef, tag, nb, info_dict)."""
        R, ns, init_ecef = prepare_process_epoch(self, obs, sat, obs_sd)
        info = make_epoch_diagnostics(self)

        # Phase 1 ns<4: cannot form DD, just advance time
        if self.phase == 1 and ns < 4:
            return _tc_recovery.advance_epoch_and_pack(self, self.nav.x[0:3], 'FLT', 0, info, obs)

        if self.phase == 1:
            return self._run_init_epoch(
                obs, obsb, rs, vs, dts, rsb, sat, el, iu, obs_sd, ir_map,
                info, init_ecef, R)
        return self._run_tc_epoch(
            obs, obsb, rs, vs, dts, rsb, sat, el, iu, obs_sd, ir_map,
            ref_ecef, info, ns, init_ecef, R)


    _make_isam2 = staticmethod(_tc_isam.make_isam2)
    process_imu_only = _tc_recovery.process_imu_only
    _run_init_epoch = _initialization.run_init_epoch
    _run_tc_epoch = _pipeline.run_tc_epoch


def _install_forwarders(cls):
    for name, view_attr in cls._VIEW_FORWARDS.items():
        def getter(self, _v=view_attr):
            return getattr(self, _v)
        def setter(self, value, _v=view_attr):
            self._assign_view(getattr(self, _v), value)
        setattr(cls, name, property(getter, setter))
    for name, (group, field) in cls._FIELD_FORWARDS.items():
        def getter(self, _g=group, _f=field):
            return getattr(getattr(self, _g), _f)
        def setter(self, value, _g=group, _f=field):
            setattr(getattr(self, _g), _f, value)
        setattr(cls, name, property(getter, setter))
    return cls


_install_forwarders(ImuGnssTc)
