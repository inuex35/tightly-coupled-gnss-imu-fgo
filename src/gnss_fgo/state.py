"""TC pipeline state — enum + implicit-flag aggregation point.

The TC pipeline maintains several pieces of cross-cutting state that
were historically scattered as ad-hoc flags on ImuGnssTc. This module
collects them in one place so the eventual 5-state FSM (NORMAL ↔
DEGRADED ↔ DR ↔ INS_ONLY ↔ RECOVERY) has a single landing point.

Currently, only the cp-hold trigger lives here as a function; the
flags themselves remain on the runner instance (see "Aggregated flags"
below). They are migrated incrementally — keeping the bit-exact
verification trail short — and will move into a TcState dataclass in
a follow-up commit when the FSM lands.

Aggregated flags (still on tc.* instance):

  * ``skip_count``           consecutive GDOP-skipped epochs
  * ``_recov_cp_hold``       remaining DDCP-disabled epochs after a trigger
  * ``_pim_discontinuity``   one-shot signal: break the IMU chain at next epoch
  * ``_ddpr_bad_count``      consecutive epochs above main-DDPR-residual gate

Future SOTA hooks: TcState enum + transition table (recovery actions
register here), TcScene preset for scene-aware tuning.
"""

from enum import Enum



class TcState(Enum):
    """Five-state FSM landing point. Not yet wired into the runner —
    transitions still happen via the implicit flags above.
    """
    NORMAL    = 'normal'      # Fix flowing, residuals clean.
    DEGRADED  = 'degraded'    # Some sat outliers / CP-hold engaged.
    DR        = 'dr'          # Dead reckoning — GDOP skip, IMU-only update.
    INS_ONLY  = 'ins_only'    # Long outage, FLS frozen.
    RECOVERY  = 'recovery'    # Just warm-reset, gating new measurements.


def trigger_cp_hold(tc, reason, info, value=None, skip_if_active=False):
    """Engage global CP-hold for RECOV_CP_HOLD epochs.
    Triggers: slip_burst (≥N sats slipped this epoch),
    innovation (pose jump from IMU prediction), fde_safeguard (runaway FDE).

    skip_if_active=True prevents re-trigger during active hold (fde/innovation
    would fire every epoch during recovery, creating infinite loop).
    """
    if skip_if_active and tc._recov_cp_hold > 0:
        return False
    hold_n = effective_cp_hold_epochs(tc)
    tc._recov_cp_hold = max(tc._recov_cp_hold, hold_n)
    tc._recov_cp_release_streak = 0
    sq = getattr(tc, '_sat_quality', None)
    if sq is not None:
        sq.clear()
    info[f'cp_hold_{reason}'] = value if value is not None else True
    return True


def effective_cp_hold_epochs(tc) -> int:
    """Configured cp-hold length, with startup bootstrap suppression.

    During the first few live Phase-2 epochs we rely on the DDPR-only
    translation anchor to hand off from the init graph into the regular
    graph. A global CP hold during that same window leaves the graph
    floating with no DDCP/AR pull-back channel, so suppress it until the
    bootstrap-DDPR countdown expires.
    """
    if int(tc._tc_bootstrap_ddpr_epochs or 0) > 0:
        return 0
    return int(tc.cfg.recov_cp_hold)
