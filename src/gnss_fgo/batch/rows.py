"""Column layout of the measurement arrays produced by batch.frontend.

DD rows (one per epoch, system, frequency and target satellite; DD of a
quantity Q is (Q_rt - Q_bt) - (Q_rr - Q_br), so the DD ambiguity is
N_tgt - N_ref of the between-receiver SD arcs):
"""
import gtsam

EPOCH, FREQ, LAM, WEIGHT, ARC_REF, ARC_TGT = 0, 1, 2, 3, 4, 5
PR_RR, PR_BR, PR_RT, PR_BT = 6, 7, 8, 9          # pseudoranges [m]: rover/base x ref/tgt
CP_RR, CP_BR, CP_RT, CP_BT = 10, 11, 12, 13      # carriers [m]
SAT_RR, SAT_RT, SAT_BR, SAT_BT = (slice(14, 17), slice(17, 20),   # sat ECEF at rover
                                  slice(20, 23), slice(23, 26))   # / base signal time
SATNO_REF, SATNO_TGT = 26, 27
N_DD_COLS = 28

# Rover Doppler rows (first non-zero band per satellite):
D_EPOCH, D_SAT, D_HZ, D_LAM, D_EL = 0, 1, 2, 3, 4
D_SATPOS, D_SATVEL = slice(5, 8), slice(8, 11)
N_DOP_COLS = 11


def sat_points(r):
    """(ref@rover, tgt@rover, ref@base, tgt@base) satellite positions."""
    return (gtsam.Point3(*r[SAT_RR]), gtsam.Point3(*r[SAT_RT]),
            gtsam.Point3(*r[SAT_BR]), gtsam.Point3(*r[SAT_BT]))
