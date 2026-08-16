"""TcConfig — tunables for ImuGnssTc. Flat (`cfg.zupt_max_acc_std`) and namespaced (`cfg.zupt.max_acc_std`) access are equivalent."""

import os
from dataclasses import dataclass

from .utils import env_f, env_i


class _SubConfigView:
    """Prefix-stripped namespace proxy for a group of TcConfig fields."""
    __slots__ = ('_cfg', '_prefix')

    def __init__(self, cfg, prefix):
        object.__setattr__(self, '_cfg', cfg)
        object.__setattr__(self, '_prefix', prefix)

    def __getattr__(self, name):
        return getattr(self._cfg, f'{self._prefix}_{name}')

    def __setattr__(self, name, value):
        setattr(self._cfg, f'{self._prefix}_{name}', value)

    def __repr__(self):
        keys = [k[len(self._prefix) + 1:] for k in dir(self._cfg)
                if k.startswith(self._prefix + '_')
                and not k.startswith('_')]
        kv = ', '.join(f'{k}={getattr(self, k)!r}' for k in sorted(keys))
        return f'{self._prefix}Config({kv})'


IMU_PRESETS = {
    'tactical':   dict(accel_noise=2.84e-4, gyro_noise=4.01e-5,
                       accel_bias_sigma=3.14e-4, gyro_bias_sigma=9.70e-6),
    'consumer':   dict(accel_noise=2.0e-3,  gyro_noise=4.0e-4,
                       accel_bias_sigma=1.0e-3,  gyro_bias_sigma=5.0e-5),
    'industrial': dict(accel_noise=6.0e-4,  gyro_noise=1.0e-4,
                       accel_bias_sigma=5.0e-4,  gyro_bias_sigma=2.0e-5),
    'nav_grade':  dict(accel_noise=5.7e-5,  gyro_noise=3.5e-6,
                       accel_bias_sigma=2.0e-5,  gyro_bias_sigma=5.0e-7),
}

TC_PRESETS = {}


@dataclass
class TcConfig:
    """All tunables for ImuGnssTc. Loaded from env vars at __init__."""
    # DD observation σ
    sigma_pr: float = 0.3          # DD pseudorange σ [m]
    sigma_cp: float = 0.003        # DD carrier phase σ [m]
    sigma_amb0: float = 30.0       # new satellite N prior σ [cyc]
    sigma_cont: float = 1.0        # continuing N prior σ [cyc]
    sigma_n_between: float = 0.01      # BetweenFactor N σ [cyc] — FIX
    sigma_n_between_flt: float = 0.1   # BetweenFactor N σ [cyc] — FLT
    sigma_n_between_warmup: int = 0
    betweenn_enable: int = 1
    fix_pose_anchor_sigma: float = 0.0
    el_mask_deg: float = 10.0
    elmin_deg: float = 15.0
    cnr_min_dbhz: float = 25.0
    snr_ref_dbhz: float = 42.0

    varerr_enable: int = 1
    err_a: float = 0.001           # base CP σ [m]                 RTKLIB err[1]=0.003
    err_b: float = 0.001           # el CP σ [m] (b/sin(el) term)  RTKLIB err[2]=0.003
    err_eratio_pr: float = 100.0   # PR/CP σ ratio                 RTKLIB eratio[0]=300
    err_sclkstab: float = 5e-12    # rcv clock stability [s/s]     RTKLIB sclkstab

    # Robust / FDE
    huber_pr: float = 0.0          # Huber threshold for DDPR (0=off).
    pr_robust_kind: str = 'huber'  # 'huber' | 'cauchy' | 'dcs' | 'tukey'
                                   # GREAT-FGO-style: cauchy with k ~2
    fde_pr: float = 4.0            # FDE DDPR residual threshold [m]
    fde_cp: float = 0.5            # FDE DDCP residual threshold [m]
    fde_max_frac: float = 0.5      # skip FDE if >this fraction rejected
    fde_enable: int = 1           # 0=off, 1=on (FDE_ENABLE)
    fde_max_iter: int = 1
    ddpr_sanity_enable: int = 1   # 0=off, 1=on (DDPR_SANITY_ENABLE)
    sanity_max_gdop: float = 0.0
    sanity_break_pim: int = 1
    sanity_pose_replace_thresh: float = 5.0
    varholdamb: float = 0.001
    cp_hold_isam_iters: int = 0
    # Held-N gauge gate [m]: drop a held N when the fresh seed disagrees
    # by more than this. Diagnostic only (0 = off) — the clock-free
    # cp-pr seed removed the gauge drift this defended against, and a
    # tight gate expires healthy holds. See buildfactor/amb_seed.py.
    hold_gauge_gate_m: float = 0.0
    pim_break_trans_sigma: float = 1.0
    # cssrlib valpos chi-square threshold in σ units.
    valpos_thres: float = 4.0

    # Cycle slip / multipath
    thres_slip: float = 0.15       # GF slip threshold [m]
    cmc_thresh: float = 3.0        # Code-minus-carrier jump threshold [m]
    cmc_level_thresh: float = 0.0
    cmc_warmup_epochs: int = 5     # avg before steady-state monitoring
    cmc_alpha: float = 0.05        # smoothing factor for steady-state baseline
    cn0_min: float = 0.0           # C/N0 floor [dB-Hz], 0=off

    thresdop: float = 0.0          # try 5–10 cyc/s on a clean run

    # Raw per-satellite Doppler (gtsam.DopplerFactorArm). σ is the range-rate
    # measurement noise; 0 disables. The clock-bias chain the factors
    # difference needs two more knobs: a random walk between epochs (as a
    # range-rate σ, converted to seconds inside the builder) and the anchor
    # prior that pins the otherwise-unobservable absolute bias.
    doppler_sigma: float = 0.0     # [m/s] 0 = off — raw, with clock states
    doppler_adaptive_sigma: int = 1  # σ follows the epoch's own residual scale
    doppler_fde_k: float = 4.0     # drop Dopplers beyond k robust scales
    doppler_snr_weight: int = 1    # scale Doppler σ by C/N0 as varerr does
    doppler_gdop_max: float = 0.0  # skip Doppler above this GDOP (0 = off)
    doppler_require_dd: int = 1    # only add Doppler where the epoch has a
                                   # usable DD set (see buildfactor/doppler_sd)
    doppler_skip_aid: int = 1      # SD Doppler also on GDOP-skipped epochs
                                   # (outage velocity aid; bypasses the
                                   # require_dd/gdop gates there)
    doppler_sd_sigma: float = 0.5  # [m/s] 0 = off — between-satellite
                                   # difference, no clock states. 0.5 is
                                   # the measured full-length optimum.
    doppler_huber: float = 1.0     # [m/s] robust width, 0 = plain L2.
                                   # Load-bearing: bounds the NLOS
                                   # feedback loop in the SD screen (see
                                   # buildfactor/doppler_sd.py).
    doppler_clk_rw: float = 100.0  # [m/s] clock-drift random walk per epoch
    clock_pr_anchor_sigma: float = 1e-6   # [s] loose anchor on a chain head
                                   # when pseudoranges observe the level
    clock_pr_sigma: float = 0.0    # [m] zenith σ of the undifferenced GPS
                                   # iono-free pseudoranges that observe the
                                   # clock chain (0 = off; needs doppler_sigma)
    doppler_clk_anchor_sigma: float = 3.3e-9  # [s] prior pinning the head of
                                   # a clock chain (the level is unobservable
                                   # from range rates, so any value does)
    tdcp_sigma: float = 0.0        # [m] TDCP σ between consecutive poses;
                                   # 0 disables (default). Experimental:
                                   # cancels ambiguity/slow biases and
                                   # bounds float drift in NLOS storms.
                                   # tokyo run2 measurements: huber kernel
                                   # -> best FixRMS (0.238) but a 4 km
                                   # mass-slip excursion; tukey -> best
                                   # AllRMS (11.8) but AR dies. Needs a
                                   # kernel/sigma sweep before default-on.

    mw_thresh: float = 0.0
    mw_avg_enable: int = 1
    gf_avg_enable: int = 0

    # GNSS quality gate
    gdop_max: float = 10.0
    nsat_min: int = 6
    min_dd_for_solve: int = 4
    propagate_pose_sigma: float = 1.0   # m — IMU-pred pose prior σ (translation)
    propagate_vel_sigma:  float = 1.0   # m/s — IMU-pred velocity prior σ
    propagate_bias_sigma: float = 0.1   # IMU-pred bias prior σ
    propagate_amb_sigma:  float = 0.1   # cycles — IMU-pred N prior σ

    # CP-hold triggers
    recov_cp_hold: int = 5         # hold DDCP for N epochs after any trigger
    recov_cp_release_thresh: float = 0.0
    recov_cp_release_count: int = 3
    bad_sat_release_thresh_scale: float = 0.7
    bad_sat_release_count_scale: float = 2.0
    cp_hold_sigma_penalty: float = 0.0
    ddcp_res_weight_thresh_m: float = 0.0
    ddcp_res_weight_stale_max_epochs: int = 2
    sanity_max_median_ratio: float = 5.0
    sanity_max_median_min_sats: int = 6
    ddcp_res_weight_max_m: float = 0.0
    imu_integ_cov_max: float = 0.5
    obsq_res_thresh: float = 2.0
    obsq_bad_streak_cap: int = 8
    obsq_release_thresh_scale: float = 0.85
    obsq_release_count_scale: float = 1.5

    main_ddpr_res_thresh: float = 3.0
    main_ddpr_per_sat_thresh: float = 0.0
    post_ar_cost_thresh: float = 9999.0
    ddpr_sanity_persist: int = 3      # 3 consecutive bad → DDPR-LS anchor
    ddpr_max_res: float = 2.0
    anchor_imu_max_gap: float = 20.0
    anchor_imu_hard_max: float = 200.0
    anchor_imu_clean_res: float = 1.0          # m, DDPR-LS self-residual
    anchor_imu_clean_main_res: float = 15.0    # m, post-fit DDPR RMS
    main_ddpr_res_catastrophic: float = 15.0
    ar_ddpr_xvalidate_thresh: float = 10.0
    ar_ddpr_xvalidate_delta_thresh: float = 0.0
    ddpr_fast_worst_sat_min: float = 30.0
    ddpr_bad_persist_override: int = 6
    ddpr_clean_res: float = 1.0
    per_sat_res_thresh: float = 3.0

    # IMU noise (select via imu_grade or override individual σ)
    imu_grade: str = 'tactical'    # 'tactical'|'consumer'|'industrial'|'nav_grade'
    imu_scale: float = 1.0         # multiplier on accel/gyro noise +
    imu_integ_cov: float = 1e-3    # position integration noise [m²/s]
    accel_noise: float = 2.84e-2   # [m/s²/√Hz]   (spec ×100)
    gyro_noise: float = 4.01e-2    # [rad/s/√Hz]  (spec ×1000) — looser
    accel_bias_sigma: float = 3.14e-4
    gyro_bias_sigma: float = 9.70e-6

    bias_between_acc_sigma:  float = 3e-4    # m/s² (spec×1)
    bias_between_gyro_sigma: float = 3e-5    # rad/s
    bias_prior_acc_sigma:    float = 3e-3    # m/s²
    bias_prior_gyro_sigma:   float = 3e-4    # rad/s ≈ 0.017 deg/s
    bias_prior_mode:         int = 2         # 0=off, 1=phase1 init, 2=prev/current
    init_pitch_deg:          float = float('nan')
    # AR
    ar_mode: int = 3               # 0=none, 1=cont, 3=fix-and-hold
    ar_max_frac: float = 1.0       # skip AR if max DD float fraction > this [cyc]
    ar_wait_new: int = 3           # new amb waits N epochs before AR
    parmode: int = 1
    par_P0: float = 0.995          # PAR success-rate threshold (parmode=2 only)
    ar_starve_reset: int = 50      # epochs of consecutive lambda_zero
                                   # (ratio starvation) with a QUIET
                                   # float that trigger the ambiguity
                                   # purge (reset_ambiguities_with_
                                   # cp_hold). A biased-but-smooth float
                                   # basin passes no acceptance test and
                                   # never trips the residual/innovation
                                   # alarms — this is its dedicated
                                   # escape. 0 = off.
    ar_starve_max_res: float = 2.0 # 'quiet' gate [m]: skip the purge
                                   # when main DDPR res exceeds this
                                   # (an NLOS storm, where purging arcs
                                   # would destroy the CP continuity
                                   # that bounds float drift)
    ar_gdop_max: float = 8.0       # skip the AR attempt when GDOP
                                   # exceeds this (0 = off). Pure
                                   # geometry — unlike a covariance
                                   # gate it has no feedback loop with
                                   # the hold state. Weak geometry
                                   # cannot support an integer decision
                                   # (9 m vertical basin at GDOP~10
                                   # costs only ~1.7 m code residual).
    ar_fix_dres_max: float = 1.0   # [m] likelihood-ratio fix gate in
                                   # the graph's own objective: reject
                                   # when DDPR RMS at the fixed pose
                                   # exceeds the float-pose RMS by more
                                   # than this (0 = off). Differential,
                                   # so the NLOS noise floor cancels;
                                   # evaluated pre-hold so wrong-integer
                                   # basins are rejected before holds
                                   # lock them.
    ar_thresar: float = 3.0        # nav.thresar — ratio gate for parmode=1
    rtklib_mode: int = 1
    ar_arfilter: int = 1           # demote newly-acquired sats hurting ratio
    ar_native_resolver: int = 0    # 1 = AR off the smoother (gnss_fgo.ar).
                                   # Line-identical to cssrlib over the FIRST
                                   # 3000 tokyo run2 epochs and shadow-equal
                                   # per call, but the full 9151-epoch run
                                   # measures 42.6 m / 5670 fix against
                                   # 30.6 m / 5818 on the cssrlib path -- the
                                   # equivalence does not yet cover the
                                   # warm-reset-heavy tail, so the proven
                                   # path stays default until it does.
    ar_minfixsats: int = 4         # min sats to attempt AR (after exclusion)
    subset_ar_enable: int = 1
    subset_ar_max_candidates: int = 5
    subset_ar_min_nb: int = 4
    subset_ar_max_drop: int = 2
    subset_ar_max_mres_m: float = 0.0
    subset_ar_max_dirty_sats: int = 2
    subset_ar_dirty_sat_res_m: float = 1.0
    exclude_bds_geo: int = 1  # BeiDou-2 GEO broadcast orbits are hundreds-
                              # of-metres class with heavy stationary-
                              # geometry code multipath; standard RTK
                              # practice excludes C01-C05/C59-C63
    ar_context_main_ddpr_max: float = 1.2
    ar_context_worst_sat_max: float = 4.0
    ar_context_nb_max: int = 6
    ar_context_reject_during_cp_hold: int = 1
    ar_context_reject_during_ddpr_bad: int = 1
    ar_precheck_skip: int = 0
    ar_min_nb: int = 0
    lambda_corr_max: float = 0.0
    weak_fix_nb_max: int = 2
    weak_fix_lambda_corr_max: float = 0.08
    weak_fix_main_ddpr_res_max: float = 0.8
    weak_fix_only_after_flt: int = 1
    weak_fix_reject_max_prev_fix_streak: int = 2
    low_nb_fix_reject_nb_max: int = 6
    low_nb_fix_only_after_flt: int = 1
    low_nb_fix_reject_max_prev_fix_streak: int = 2
    lambda_corr_hard_max: float = 1.0
    diag_truth_residual: int = 0
    diag_main_ddpr_res: int = 1
    # Per-bucket factor RMS dump (info['fres_*'] / info['fcnt_*']). Default
    # off because evaluating every Python CustomFactor per epoch costs ~10%
    # of wall time on the tokyo run; turn on only for offline diagnostics.
    diag_factor_residuals: int = 0
    fls_update_timing: int = 0
    ar_persist_bad_enable: int = 1
    ar_persist_bad_res_thresh: float = 2.0
    ar_persist_bad_streak: int = 4
    ar_persist_bad_hold: int = 10

    # Phase transition
    vel_thresh: float = 1.0        # motion detection for Phase 1→2 [m/s]
    n_collect: int = 5             # Fix positions collected before Phase 2
    body_rot_std: float = 1.0      # HMC heading uncertainty [deg]

    nhc_enable: int = 1            # 1=on, 0=off
    nhc_min_speed: float = 0.0     # engage even at stop (body-lateral ≈ 0 holds)
    nhc_sigma_lat: float = 0.3     # lateral velocity σ [m/s] — tight enough
    nhc_sigma_vert: float = 0.2    # vertical velocity σ [m/s]
    nhc_lever: str = '0,0,0'

    zupt_enable: int = 1
    zupt_max_acc_std: float = 0.55          # m/s² (engine vibration var)
    zupt_max_gyro_std: float = 0.030        # rad/s (vibration var)
    zupt_max_gyro_median: float = 0.020     # rad/s (rate magnitude post-bias)
    zupt_g_dev_thr: float = 0.0             # m/s² (0 disables grav check)
    zupt_max_speed: float = 0.0             # m/s (vel-prev gate, 0 disables)
    zupt_min_samples: int = 5
    zupt_sigma_zero_velocity: float = 0.5   # m/s   — ZUPT prior σ
    zupt_sigma_zero_rotation: float = 0.010 # rad   — ZARU between-factor σ
    zupt_anchor_sigma_translation: float = 1.0  # m   — anchor PriorPose3 σ
    zupt_anchor_sigma_rotation: float = 0.1     # rad — anchor PriorPose3 σ

    # FLS
    fls_lag: float = 1.0           # FixedLagSmoother window [s]
    phase1_fls_lag: float = 1.0    # Phase 1 lag [s] — re-stamp keys each epoch keeps active set bounded
    isam2_relinearize_skip: int = 10
    isam2_relinearize_threshold: float = 0.05

    def __post_init__(self):
        """Apply IMU grade preset if user hasn't explicitly set individual σ."""
        tactical = IMU_PRESETS['tactical']
        is_default = (self.accel_noise == tactical['accel_noise'] and
                      self.gyro_noise == tactical['gyro_noise'] and
                      self.accel_bias_sigma == tactical['accel_bias_sigma'] and
                      self.gyro_bias_sigma == tactical['gyro_bias_sigma'])
        if is_default and self.imu_grade != 'tactical':
            preset = IMU_PRESETS.get(self.imu_grade)
            if preset is None:
                raise ValueError(f"Unknown imu_grade: {self.imu_grade}. "
                                 f"Choose from {list(IMU_PRESETS)}")
            for k, v in preset.items():
                setattr(self, k, v)

    _ENV_OVERRIDES = {
        'sigma_pr':                'SIG_PR',
        'sigma_cp':                'SIG_CP',
        'sigma_amb0':              'SIG_AMB',
        'sigma_cont':              'SIG_CONT',
        'sigma_n_between':         'SIG_N_BETWEEN',
        'sigma_n_between_flt':     'SIG_N_BETWEEN_FLT',
        'sigma_n_between_warmup':  'SIG_N_BETWEEN_WARMUP',
        'thres_slip':              'THRESSLIP',
        'zupt_g_dev_thr':          'ZUPT_G_DEV',
    }

    @classmethod
    def _env_name(cls, field_name):
        return cls._ENV_OVERRIDES.get(field_name, field_name.upper())


    @property
    def zupt(self):    return _SubConfigView(self, 'zupt')

    @classmethod
    def from_env(cls):
        kw = {}
        preset_name = os.environ.get('TC_PRESET', '').strip()
        preset = TC_PRESETS.get(preset_name, {})
        for name, f in cls.__dataclass_fields__.items():
            if name == 'imu_grade':
                continue   # handled below
            envname = cls._env_name(name)
            default = preset.get(name, f.default)
            if isinstance(f.default, str):
                kw[name] = os.environ.get(envname, default)
            elif isinstance(f.default, int) and not isinstance(f.default, bool):
                kw[name] = env_i(envname, default)
            else:
                kw[name] = env_f(envname, default)
        # String field: imu_grade
        kw['imu_grade'] = os.environ.get('IMU_GRADE', cls.imu_grade)
        return cls(**kw)
