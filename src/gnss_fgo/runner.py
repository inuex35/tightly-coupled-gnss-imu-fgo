"""ImuGnssTc — single-phase IMU/GNSS tightly-coupled FGO runner."""

import numpy as np
import gtsam
from cssrlib.rtk import rtkpos
from cssrlib.gnss import ecef2pos, timediff

from .utils import (
    estimate_stationary_bias as _utils_estimate_stationary_bias,
    make_imu_params as _utils_make_imu_params,
)
from .config import TcConfig
from . import pipeline as _pipeline
from .integrity import recovery as _tc_recovery
from .state.runtime_state import CurrentEpochState, SatStateMap
from .pipeline import update_smoother as _tc_isam


class ImuGnssTc:
    """Single-phase IMU/GNSS tightly-coupled factor-graph processor."""

    Xpose = staticmethod(lambda i: gtsam.symbol('P', i))
    Vel = staticmethod(lambda i: gtsam.symbol('v', i))
    Bias = staticmethod(lambda i: gtsam.symbol('b', i))
    N = staticmethod(lambda s, f, gen=0: gtsam.symbol(
        'n', int(gen) * 1000000 + int(s) * 10 + int(f)))

    # ── cssrlib boundary ──────────────────────────────────────────
    def prepare_double_difference_measurements(self, *a, **k):
        return self.engine.prepare_double_difference_measurements(*a, **k)

    def IB(self, *a, **k):
        return self.engine.IB(*a, **k)

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
        self.nav.armode = int(self.cfg.armode)
        self.nav.elmin = np.deg2rad(float(self.cfg.elmin_deg))
        self.nav.cnr_min = float(self.cfg.cnr_min_dbhz)
        self.nav.thresar = 3.0
        self.nav.minfixsats = 4
        self.nav.sat_band_plan = True

    def _init_runtime_state(self):
        self.epoch = 0
        self.imu_idx = 0
        self.tc_epoch = 0

        self._sat_states = SatStateMap()

        self.isam2 = None
        self.imu_params = None
        self.tc_bias = None
        self.lever_arm_tc = gtsam.Point3(0, 0, 0)

        self._last_obs_t = None
        self._epoch_dt = 0.2
        self.current_epoch = CurrentEpochState()
        self._nhc_lever_cache = None
        self._nhc_lever_src = None
        self._ar_key_pose = None
        self.fls_lag = None
        self.tc_bias_init = None
        self._ar_cp_visible_sf = set()

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

    def _make_imu_params(self):
        return _utils_make_imu_params()

    def _antenna_ecef(self, pose, ecef_body):
        lever_arr = np.array(self.lever_arm_tc)
        if np.linalg.norm(lever_arr) == 0:
            return ecef_body
        R_body = self.ecef_T_nav.compose(pose).rotation().matrix()
        return ecef_body + R_body @ lever_arr

    def _estimate_stationary_bias(self):
        return _utils_estimate_stationary_bias(
            self.imu_data, n_max=min(3000, self.imu_idx))

    def process(self, obs, obsb, rs, vs, rsb, sat, el, iu, obs_sd, ir_map):
        self._update_epoch_dt(obs)
        self._reset_current_epoch()
        R, ns, init_ecef = self.R_enu2ecef, len(sat), self.nav.x[0:3].copy()
        info = {'tc_epoch': self.tc_epoch}

        if ns < 4:
            return _tc_recovery.advance_epoch_and_pack(
                self, self.nav.x[0:3], 'FLT', 0, info, obs)

        return self._run_tc_epoch(
            obs, obsb, rs, vs, rsb, sat, el, iu, obs_sd, ir_map,
            info, ns, init_ecef, R)

    _make_isam2 = staticmethod(_tc_isam.make_isam2)
    process_imu_only = _tc_recovery.process_imu_only
    _run_tc_epoch = _pipeline.run_tc_epoch
