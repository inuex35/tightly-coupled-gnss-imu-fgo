"""ImuGnssTc — clean two-phase IMU/GNSS tightly-coupled FGO runner."""

import numpy as np
import gtsam
from cssrlib.rtk import rtkpos
from cssrlib.gnss import ecef2pos, timediff

from .utils import (
    collect_imu_samples as _utils_collect_imu_samples,
    build_pim_from_samples as _utils_build_pim_from_samples,
    estimate_stationary_bias as _utils_estimate_stationary_bias,
    DDPRContext as _DDPRContext,
    ddpr_only_position as _utils_ddpr_only_position,
    heading_from_vel as _utils_heading_from_vel,
    make_imu_params as _utils_make_imu_params,
)
from .config import TcConfig
from .factors.epoch_context import make_epoch_diagnostics, prepare_process_epoch
from . import initialization as _initialization
from . import pipeline as _pipeline
from .integrity import recovery as _tc_recovery
from .state.runtime_state import (
    ArDiagnostics, CurrentEpochState, SatStateMap,
)
from .pipeline import update_smoother as _tc_isam
from .factors import prefit as _tc_prefit


class ImuGnssTc:
    """Two-phase IMU/GNSS tightly-coupled factor-graph processor."""

    Xp = staticmethod(lambda i: gtsam.symbol('x', i))
    Xpose = staticmethod(lambda i: gtsam.symbol('P', i))
    Vel = staticmethod(lambda i: gtsam.symbol('v', i))
    Bias = staticmethod(lambda i: gtsam.symbol('b', i))
    N = staticmethod(lambda s, f, gen=0: gtsam.symbol(
        'n', int(gen) * 1000000 + int(s) * 10 + int(f)))

    # ── cssrlib boundary ──────────────────────────────────────────
    def prepare_double_difference_measurements(self, *a, **k):
        return self.engine.prepare_double_difference_measurements(*a, **k)

    def zdres(self, *a, **k):
        return self.engine.zdres(*a, **k)

    def sdres(self, *a, **k):
        return self.engine.sdres(*a, **k)

    def ddidx(self, *a, **k):
        return self.engine.ddidx(*a, **k)

    def holdamb_flags(self, *a, **k):
        return self.engine.holdamb_flags(*a, **k)

    def IB(self, *a, **k):
        return self.engine.IB(*a, **k)

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

    @staticmethod
    def _noise1(sigma):
        return gtsam.noiseModel.Isotropic.Sigma(1, float(sigma))

    def __init__(self, nav, pos0, base_ecef, imu_data,
                 lever_arm=np.zeros(3), logfile=None, cfg=None):
        self.engine = rtkpos(nav, pos0, logfile)
        self.nav = self.engine.nav

        self.base_ecef = np.array(base_ecef)
        self.imu_data = imu_data
        self.lever_arm = np.array(lever_arm, dtype=float)

        self._init_base_frame()

        self.cfg = cfg if cfg is not None else TcConfig.from_env()
        self._apply_nav_config()
        self._init_runtime_state()

    def _init_base_frame(self):
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
        self.nav.armode = 1
        self.nav.elmin = np.deg2rad(float(self.cfg.elmin_deg))
        self.nav.cnr_min = float(self.cfg.cnr_min_dbhz)
        self.nav.thresar = 3.0
        self.nav.minfixsats = 4
        self.nav.sat_band_plan = True

    def _init_runtime_state(self):
        self.phase = 1
        self.epoch = 0
        self.imu_idx = 0
        self.tc_epoch = 0

        self.phase1_t = 0.0
        self.isam = self._make_isam2(self.cfg.fls_lag)
        self.amb_keys = {}
        self._isam_p1_inserted = set()

        self.collecting = False
        self.collected_fixes = []

        self._sat_states = SatStateMap()

        self.isam2 = None
        self.imu_params = None
        self.tc_bias = None
        self.lever_arm_tc = gtsam.Point3(0, 0, 0)

        self._last_obs_t = None
        self._epoch_dt = 0.2
        self.current_epoch = CurrentEpochState()
        self.ar_diag = ArDiagnostics()
        self._last_ar_result = None
        self._nhc_lever_cache = None
        self._nhc_lever_src = None
        self._ar_key_pose = None
        self._sol_hist = []
        self.fls_lag = None
        self.tc_bias_init = None
        self._ar_cp_visible_sf = set()
        self._last_sol_ecef = None
        self._last_s0 = 0.0
        self._last_s1 = 0.0

    def _update_epoch_dt(self, obs):
        if self._last_obs_t is not None:
            try:
                dt = float(timediff(obs.t, self._last_obs_t))
                if dt > 0:
                    self._epoch_dt = dt
            except (TypeError, ValueError):
                pass
        self._last_obs_t = obs.t

    def _reset_current_epoch(self):
        """Replace epoch-scoped scratch; keep persistent arc amb_gen monotonic."""
        self.current_epoch = CurrentEpochState()
        self.ar_diag = ArDiagnostics()

    def _make_imu_params(self):
        return _utils_make_imu_params()

    def _antenna_ecef(self, pose, ecef_body):
        lever_arr = np.array(self.lever_arm_tc)
        if np.linalg.norm(lever_arr) == 0:
            return ecef_body
        R_body = self.ecef_T_nav.compose(pose).rotation().matrix()
        return ecef_body + R_body @ lever_arr

    def _collect_imu_samples(self, n_samples=100, target_tow=None):
        samples, self.imu_idx = _utils_collect_imu_samples(
            self.imu_data, self.imu_idx,
            n_samples=n_samples, target_tow=target_tow)
        return samples

    def _estimate_stationary_bias(self):
        return _utils_estimate_stationary_bias(
            self.imu_data, n_max=min(3000, self.imu_idx))

    def _build_pim_from_samples(self, bias, imu_samples):
        return _utils_build_pim_from_samples(self.imu_params, bias, imu_samples)

    def _heading_from_vel(self, vel, fallback, disp_enu=None):
        return _utils_heading_from_vel(vel, fallback, disp_enu)

    def _ddpr_only_position(self, obs, obsb, obs_sd, rs, rsb,
                            sat, el, iu, ir_map, pose_init):
        sigma_pr_zenith = _tc_prefit.varerr_dd_sigma(self, 1, np.pi / 2, self._epoch_dt)
        ctx = _DDPRContext(
            R_enu2ecef=self.R_enu2ecef,
            base_ecef=self.base_ecef,
            base_pt=self.base_pt,
            ecef_T_nav=self.ecef_T_nav,
            lever=self.lever_arm_tc,
            nav_nf=self.nav.nf,
            sigma_pr=sigma_pr_zenith / np.sqrt(2.0),
            fde_pr=4.0,
            pick_ref_sat_idx=lambda *a, **k: _tc_prefit.pick_ref_sat_idx(self, *a, **k),
        )
        ecef_out, n_active, res_rms, _, _ = _utils_ddpr_only_position(
            obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
            pose_init, ctx)
        return ecef_out, n_active, res_rms

    def process(self, obs, obsb, rs, vs, rsb, sat, el, iu, obs_sd, ir_map):
        R, ns, init_ecef = prepare_process_epoch(self, obs, sat)
        info = make_epoch_diagnostics(self)

        if self.phase == 1 and ns < 4:
            return _tc_recovery.advance_epoch_and_pack(
                self, self.nav.x[0:3], 'FLT', 0, info, obs)

        if self.phase == 1:
            return self._run_init_epoch(
                obs, obsb, rs, vs, rsb, sat, el, iu, obs_sd, ir_map,
                info, init_ecef, R)
        return self._run_tc_epoch(
            obs, obsb, rs, vs, rsb, sat, el, iu, obs_sd, ir_map,
            info, ns, init_ecef, R)

    _make_isam2 = staticmethod(_tc_isam.make_isam2)
    process_imu_only = _tc_recovery.process_imu_only
    _run_init_epoch = _initialization.run_init_epoch
    _run_tc_epoch = _pipeline.run_tc_epoch
