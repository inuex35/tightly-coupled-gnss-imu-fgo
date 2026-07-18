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
from .buildfactor.epoch import make_epoch_diagnostics, prepare_process_epoch
from .buildfactor import factors as _tc_factors
from . import initialization as _initialization
from . import tightly_coupled as _tightly_coupled
from .validation import recovery as _tc_recovery
from .runtime_state import (
    AmbiguityState, MresSignalsState, RecoveryState, SatFieldView, SatStateMap,
)
from .preprocess.sat_quality import SatQualityState
from .optimize import solver as _tc_solver
from .preprocess import prefit as _tc_prefit
from .utils import sorted_sys_ids


class ImuGnssTc(rtkpos):
    """Two-phase IMU/GNSS tight coupling processor."""

    # Symbol helpers
    Xp = staticmethod(lambda i: gtsam.symbol('x', i))   # Phase 1 pose
    Xpose = staticmethod(lambda i: gtsam.symbol('P', i)) # Phase 2 pose
    Vel = staticmethod(lambda i: gtsam.symbol('v', i))
    Bias = staticmethod(lambda i: gtsam.symbol('b', i))
    N = staticmethod(lambda s, f, gen=0: gtsam.symbol(
        'n', int(gen) * 1000000 + int(s) * 10 + int(f)))

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
        super().__init__(nav, pos0, logfile)

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
        self._ambiguity = AmbiguityState()
        self._recovery = RecoveryState()
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
        self._amb_factor_indices_view = SatFieldView(
            self._sat_states, 'amb_factor_indices', absent=[])
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
        self._last_rejc_wipe = 0
        self._last_ddpr_sat_tags = []
        self._last_main_ddpr_res = 0.0
        self._last_main_ddpr_per_sat = {}
        self._last_main_ddpr_epoch = -10**9
        self._last_per_sat_res = {}
        self._last_pair_bad_max = 0.0
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
        """Refresh self._epoch_dt with elapsed seconds since the previous"""
        if self._last_obs_t is not None:
            try:
                dt = float(timediff(obs.t, self._last_obs_t))
                if dt > 0:
                    self._epoch_dt = dt
            except (TypeError, ValueError):
                pass
        self._last_obs_t = obs.t
        self.thresslip = self.cfg.thres_slip
        self.cmc_thresh = self.cfg.cmc_thresh
        self.cn0_min = self.cfg.cn0_min

        # Reference satellite tracking per system
        self.ref_sats = {}

        self.amb_gen = {}
        # Wavelength cache: {(sat, f): wavelength}
        self.amb_lam = {}

        self.amb_init_epoch = {}
        self._sat_quality = SatQualityState()
        self.ar_wait_new = self.cfg.ar_wait_new

        self.amb_factor_indices = {}
        self.total_factor_count = 0  # running count of factors added to ISAM2


    def _assign_view(self, view: SatFieldView, value):
        if value is view:
            return
        view.clear()
        if value:
            for k, v in value.items():
                view[k] = v

    @property
    def amb_keys_tc(self):
        return self._amb_key_view

    @amb_keys_tc.setter
    def amb_keys_tc(self, value):
        self._assign_view(self._amb_key_view, value)

    @property
    def amb_gen(self):
        return self._amb_gen_view

    @amb_gen.setter
    def amb_gen(self, value):
        self._assign_view(self._amb_gen_view, value)

    @property
    def amb_lam(self):
        return self._amb_lam_view

    @amb_lam.setter
    def amb_lam(self, value):
        self._assign_view(self._amb_lam_view, value)

    @property
    def amb_init_epoch(self):
        return self._amb_init_epoch_view

    @amb_init_epoch.setter
    def amb_init_epoch(self, value):
        self._assign_view(self._amb_init_epoch_view, value)

    @property
    def amb_factor_indices(self):
        return self._amb_factor_indices_view

    @amb_factor_indices.setter
    def amb_factor_indices(self, value):
        self._assign_view(self._amb_factor_indices_view, value)

    @property
    def rejc_cp_pr(self):
        return self._rejc_cp_pr_view

    @rejc_cp_pr.setter
    def rejc_cp_pr(self, value):
        self._assign_view(self._rejc_cp_pr_view, value)


    @property
    def _cp_hold_streak_persat(self):
        return self._cp_hold_streak_view

    @_cp_hold_streak_persat.setter
    def _cp_hold_streak_persat(self, value):
        self._assign_view(self._cp_hold_streak_view, value)

    @property
    def _fix_streak(self):
        return self._fix_streak_view

    @_fix_streak.setter
    def _fix_streak(self, value):
        self._assign_view(self._fix_streak_view, value)

    @property
    def total_factor_count(self):
        return self._ambiguity.total_factor_count

    @total_factor_count.setter
    def total_factor_count(self, value):
        self._ambiguity.total_factor_count = value

    @property
    def skip_count(self):
        return self._recovery.skip_count

    @skip_count.setter
    def skip_count(self, value):
        self._recovery.skip_count = value

    @property
    def _recov_cp_hold(self):
        return self._recovery.recov_cp_hold

    @_recov_cp_hold.setter
    def _recov_cp_hold(self, value):
        self._recovery.recov_cp_hold = value

    @property
    def _recov_cp_release_streak(self):
        return self._recovery.recov_cp_release_streak

    @_recov_cp_release_streak.setter
    def _recov_cp_release_streak(self, value):
        self._recovery.recov_cp_release_streak = value

    @property
    def _pim_discontinuity(self):
        return self._recovery.pim_discontinuity

    @_pim_discontinuity.setter
    def _pim_discontinuity(self, value):
        self._recovery.pim_discontinuity = value

    @property
    def _ddpr_bad_count(self):
        return self._recovery.ddpr_bad_count

    @_ddpr_bad_count.setter
    def _ddpr_bad_count(self, value):
        self._recovery.ddpr_bad_count = value

    # --- MresSignalsState shims (previous-epoch DDPR residuals) ---

    @property
    def _last_main_ddpr_res(self):
        return self._mres_signals.last_res

    @_last_main_ddpr_res.setter
    def _last_main_ddpr_res(self, value):
        self._mres_signals.last_res = float(value or 0.0)

    @property
    def _last_main_ddpr_per_sat(self):
        return self._mres_signals.per_sat

    @_last_main_ddpr_per_sat.setter
    def _last_main_ddpr_per_sat(self, value):
        self._mres_signals.per_sat = value if value else {}

    @property
    def _last_main_ddpr_epoch(self):
        return self._mres_signals.epoch

    @_last_main_ddpr_epoch.setter
    def _last_main_ddpr_epoch(self, value):
        self._mres_signals.epoch = int(value)


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

    def _compute_max_dd_frac(self, est2, obs_sd, sat, ns):
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
                if k_ref is None or not est2.exists(k_ref):
                    continue
                n_ref = est2.atDouble(k_ref)
                for ji in idx_sys:
                    js = sat[ji]
                    if js == ref_s:
                        continue
                    k_j = self._sat_states.at(js, f).amb_key
                    if k_j is None or not est2.exists(k_j):
                        continue
                    dd_cyc = n_ref - est2.atDouble(k_j)
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
                ir_map, ref_vel=None, ref_ecef=None):
        """Process one epoch. Returns (sol_ecef, tag, nb, info_dict)."""
        R, ns, init_ecef = prepare_process_epoch(self, obs, sat, obs_sd)
        info = make_epoch_diagnostics(self)

        # Phase 1 ns<4: cannot form DD, just advance time
        if self.phase == 1 and ns < 4:
            return _tc_recovery.finalize_epoch(self, self.nav.x[0:3], 'FLT', 0, info, obs)

        if self.phase == 1:
            return self._run_init_epoch(
                obs, obsb, rs, vs, dts, rsb, sat, el, iu, obs_sd, ir_map,
                info, init_ecef, R)
        return self._run_tc_epoch(
            obs, obsb, rs, vs, dts, rsb, sat, el, iu, obs_sd, ir_map,
            ref_vel, ref_ecef, info, ns, init_ecef, R)


    _make_isam2 = staticmethod(_tc_solver.make_isam2)
    process_imu_only = _tc_recovery.process_imu_only
    _run_init_epoch = _initialization.run_init_epoch
    _run_tc_epoch = _tightly_coupled.run_tc_epoch
