"""TcConfig — tunables for ImuGnssTc. Access is flat (`cfg.zupt_max_acc_std`)."""

import os
from dataclasses import dataclass

from .utils import env_f, env_i


IMU_PRESETS = {
    # 'tactical' == the class defaults below (accel/gyro noise carry the
    # measured x10/x100 inflation); listed so the table tells the truth.
    'tactical':   dict(accel_noise=2.84e-3, gyro_noise=4.01e-3,
                       accel_bias_sigma=3.14e-4, gyro_bias_sigma=9.70e-6),
    'consumer':   dict(accel_noise=2.0e-3,  gyro_noise=4.0e-4,
                       accel_bias_sigma=1.0e-3,  gyro_bias_sigma=5.0e-5),
    'industrial': dict(accel_noise=6.0e-4,  gyro_noise=1.0e-4,
                       accel_bias_sigma=5.0e-4,  gyro_bias_sigma=2.0e-5),
    'nav_grade':  dict(accel_noise=5.7e-5,  gyro_noise=3.5e-6,
                       accel_bias_sigma=2.0e-5,  gyro_bias_sigma=5.0e-7),
}


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
    betweenn_enable: int = 1
    el_mask_deg: float = 10.0
    elmin_deg: float = 15.0
    cnr_min_dbhz: float = 25.0
    snr_ref_dbhz: float = 42.0

    varerr_enable: int = 1
    err_a: float = 0.001           # base CP σ [m]                 RTKLIB err[1]=0.003
    err_b: float = 0.001           # el CP σ [m] (b/sin(el) term)  RTKLIB err[2]=0.003
    err_eratio_pr: float = 100.0   # PR/CP σ ratio                 RTKLIB eratio[0]=300
    err_sclkstab: float = 5e-12    # rcv clock stability [s/s]     RTKLIB sclkstab

    # FDE
    fde_pr: float = 4.0            # FDE DDPR residual threshold [m]
    fde_cp: float = 0.5            # FDE DDCP residual threshold [m]
    fde_max_frac: float = 0.5      # skip FDE if >this fraction rejected
    fde_enable: int = 1           # 0=off, 1=on (FDE_ENABLE)
    fde_max_iter: int = 1
    # Phase-1 FDE: the same postfit screen, run on the Phase-1 GNSS-only
    # smoother. Phase 1 used to have no residual screening at all -- the
    # admission gate was its only defence, so a contaminated cohort
    # poisoned the float, AR never fixed, and the pipeline never reached
    # Phase 2 where FDE lives (P1_FDE_ENABLE).
    p1_fde_enable: int = 1
    # Judge each satellite over the bands it transmits (cssrlib
    # nav.sat_band_plan). Off, a pre-IIF GPS (no L5) or a BeiDou-2 (B1I
    # only) fails its missing selected band every epoch and is dropped for
    # the whole session -- 19 of 47 satellites on tokyo run2. On, the
    # judgment set narrows to the bands the satellite has demonstrably
    # produced; within it the strict gate is unchanged (SAT_BAND_PLAN).
    #
    # Default ON, paired with the FFRT threshold below (best all-round
    # configuration on five of the seven measured datasets; tokyo run3
    # beats the previous defaults on every metric at once). Do not enable
    # without FFRT: alone it collapses the ratio test (tokyo run3 fix
    # 65.5% -> 14.4%). On a cssrlib build lacking nav.sat_band_plan the
    # flag is silently inert and admission stays strict.
    sat_band_plan: int = 1
    # demo5/FFRT adaptive AR ratio threshold. Equal values disable it and
    # keep the fixed ar_thresar; unequal values enable the dimension-
    # adaptive polynomial clamped to [min, max]. The known lambda_zero
    # epidemic (24% of P2 epochs at the old defaults, 45% under
    # sat_band_plan) is a fixed threshold meeting 30-50-dimensional
    # candidate sets. Default ON; the 1.5 floor is what rejects the
    # liar-hostage candidates that surface ratios of ~1.3.
    ar_thresar_min: float = 1.5
    ar_thresar_max: float = 3.0
    ddpr_sanity_enable: int = 1   # 0=off, 1=on (DDPR_SANITY_ENABLE)
    sanity_break_pim: int = 1
    sanity_pose_replace_thresh: float = 5.0
    varholdamb: float = 0.001
    pim_break_trans_sigma: float = 1.0

    # Cycle slip
    thres_slip: float = 0.15       # GF slip threshold [m]

    doppler_adaptive_sigma: int = 1  # σ follows the epoch's own residual scale
    doppler_fde_k: float = 4.0     # drop Dopplers beyond k robust scales
    doppler_snr_weight: int = 1    # scale Doppler σ by C/N0 as varerr does
    doppler_gdop_max: float = 0.0  # skip Doppler above this GDOP (0 = off)
    doppler_require_dd: int = 1    # only add Doppler where the epoch has a
                                   # usable DD set (see buildfactor/doppler_sd)
    doppler_sd_sigma: float = 0.5  # [m/s] 0 = off — between-satellite
                                   # difference, no clock states. 0.5 is
                                   # the measured full-length optimum.
    doppler_huber: float = 1.0     # [m/s] robust width, 0 = plain L2.
                                   # Load-bearing: bounds the NLOS
                                   # feedback loop in the SD screen (see
                                   # buildfactor/doppler_sd.py).

    # GNSS quality gate
    min_dd_for_solve: int = 4      # min DD FACTOR count (PR+CP; a
                                   # 3-band pair alone contributes up
                                   # to 6) below which the epoch gets
                                   # propagate priors instead
    propagate_pose_sigma: float = 1.0   # m — IMU-pred pose prior σ (translation)
    propagate_vel_sigma:  float = 1.0   # m/s — IMU-pred velocity prior σ
    propagate_bias_sigma: float = 0.1   # IMU-pred bias prior σ
    propagate_amb_sigma:  float = 0.1   # cycles — IMU-pred N prior σ

    # CP-hold triggers
    recov_cp_hold: int = 5         # hold DDCP for N epochs after any trigger
    sanity_max_median_ratio: float = 5.0
    sanity_max_median_min_sats: int = 6

    main_ddpr_res_thresh: float = 3.0   # elevation-normalized pseudo-m (see residuals.main_ddpr_residuals)
    ddpr_sanity_persist: int = 3      # 3 consecutive bad → DDPR-LS anchor
    ddpr_max_res: float = 2.0
    main_ddpr_res_catastrophic: float = 15.0   # fast-path sanity trigger
    ar_ddpr_xvalidate_thresh: float = 10.0
    ddpr_fast_worst_sat_min: float = 30.0
    per_sat_res_thresh: float = 3.0

    # IMU noise (select via imu_grade or override individual σ)
    imu_grade: str = 'tactical'    # 'tactical'|'consumer'|'industrial'|'nav_grade'
    imu_integ_cov: float = 1e-3    # position integration noise [m²/s]
    accel_noise: float = 2.84e-3   # [m/s²/√Hz]   (spec ×10)
    gyro_noise: float = 4.01e-3    # [rad/s/√Hz]  (spec ×100)
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
    ar_thresar: float = 3.0        # nav.thresar — ratio gate for parmode=1
    rtklib_mode: int = 1
    ar_arfilter: int = 1           # demote newly-acquired sats hurting ratio
    ar_minfixsats: int = 4         # min sats to attempt AR (after exclusion)
    subset_ar_enable: int = 1
    subset_ar_max_candidates: int = 5
    subset_ar_min_nb: int = 4
    subset_ar_max_drop: int = 2
    subset_ar_max_dirty_sats: int = 2
    subset_ar_dirty_sat_res_m: float = 1.0
    exclude_bds_geo: int = 1  # BeiDou-2 GEO broadcast orbits are hundreds-
                              # of-metres class with heavy stationary-
                              # geometry code multipath; standard RTK
                              # practice excludes C01-C05/C59-C63
    ar_context_main_ddpr_max: float = 1.2
    ar_context_worst_sat_max: float = 4.0
    ar_context_nb_max: int = 6
    low_nb_fix_reject_nb_max: int = 6
    low_nb_fix_only_after_flt: int = 1
    low_nb_fix_reject_max_prev_fix_streak: int = 2
    lambda_corr_hard_max: float = 1.0
    # Per-bucket factor RMS dump (info['fres_*'] / info['fcnt_*']). Default
    # off because evaluating every Python CustomFactor per epoch costs ~10%
    # of wall time on the tokyo run; turn on only for offline diagnostics.
    diag_factor_residuals: int = 0

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
        cls_fields = type(self).__dataclass_fields__
        is_default = all(
            getattr(self, k) == cls_fields[k].default
            for k in ('accel_noise', 'gyro_noise',
                      'accel_bias_sigma', 'gyro_bias_sigma'))
        if is_default:
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
        'zupt_g_dev_thr':          'ZUPT_G_DEV',
    }

    @classmethod
    def _env_name(cls, field_name):
        """Env-var name for a field: override table first, else UPPER_CASE."""
        return cls._ENV_OVERRIDES.get(field_name, field_name.upper())


    @classmethod
    def from_env(cls):
        """Build a TcConfig with every field overridable via its env var."""
        kw = {}
        for name, f in cls.__dataclass_fields__.items():
            if name == 'imu_grade':
                continue   # handled below
            envname = cls._env_name(name)
            default = f.default
            if isinstance(f.default, str):
                kw[name] = os.environ.get(envname, default)
            elif isinstance(f.default, int) and not isinstance(f.default, bool):
                kw[name] = env_i(envname, default)
            else:
                kw[name] = env_f(envname, default)
        # String field: imu_grade
        kw['imu_grade'] = os.environ.get('IMU_GRADE', cls.imu_grade)
        return cls(**kw)
