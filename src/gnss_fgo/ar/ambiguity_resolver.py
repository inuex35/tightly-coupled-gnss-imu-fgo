"""Integer ambiguity resolution over the graph's own float estimate.

Takes what the graph already has -- float ambiguities and their joint
covariance -- and returns the fixed ones. No shared state, no index
arithmetic against ``nav.na``; the ratio comes back with the result.

The double-difference construction matches cssrlib's ``ddidx``: within each
constellation and frequency, the lowest-numbered satellite above the
elevation mask is the reference and every other satellite is differenced
against it (elevations are used for masking only).
"""

from dataclasses import dataclass, field

import numpy as np
from cssrlib.gnss import sat2prn
from cssrlib.mlambda import mlambda


@dataclass
class ResolverResult:
    """Outcome of one resolution attempt.

    Deliberately NOT named ArResult: cssrlib's class of that name carries a
    boolean ``fixed`` where this one carries a dict, and the mix-up would
    only surface at the first fixing epoch. Distinct names make it an
    ImportError instead.
    """

    nb: int = 0                       # number of fixed DD ambiguities
    fixed: dict = field(default_factory=dict)   # (sat, freq) -> fixed SD value
    s0: float = 0.0
    s1: float = 0.0
    declined_partial: bool = False    # parmode-2 guard declined a partial fix
    pairs: list = field(default_factory=list)   # (ref, target, freq) per DD
    thres_used: float = 0.0           # the ratio threshold this attempt faced


class AmbiguityResolver:
    """LAMBDA over a float ambiguity vector and its covariance."""

    def __init__(self, thresar=3.0, parmode=1, par_p0=0.995, min_pairs=2,
                 el_mask=0.0, thresar_min=0.0, thresar_max=0.0):
        self.thresar = float(thresar)
        self.parmode = int(parmode)
        self.par_p0 = float(par_p0)
        self.min_pairs = int(min_pairs)
        self.el_mask = float(el_mask)
        self.thresar_min = float(thresar_min)
        self.thresar_max = float(thresar_max)

    # Polynomial coefficients for the demo5/FFRT adaptive AR ratio
    # threshold as a function of DD count (fitted to LAMBDA reliability
    # curves from TU Delft; valid for 1-50 pairs). Ported verbatim from
    # the fork's reviewed cssrlib implementation (13db2f5): rows are
    # evaluated at the base threshold first, then the threshold
    # polynomial in 1/(nb+1).
    _AR_POLY_COEFFS = (
        (-1.94058448e-01, -7.79023476e+00, 1.24231120e+02,
         -4.03126050e+02, 3.50413202e+02),
        (6.42237302e-01, -8.39813962e+00, 2.92107285e+01,
         -2.37577308e+01, -1.14307128e+00),
        (-2.22600390e-02, 3.23169103e-01, -1.39837429e+00,
         2.19282996e+00, -5.34583971e-02))

    def ratio_threshold(self, nb):
        """Ratio-test threshold for ``nb`` DD candidates.

        Fixed ``thresar`` unless ``thresar_min != thresar_max``, in which
        case the demo5 FFRT polynomial adapts it to the problem dimension
        -- a fixed ratio threshold grows ever more conservative as the
        candidate count rises (s1/s0 tends to 1 in high dimension even
        for a correct fix), which is exactly the lambda_zero epidemic the
        tokyo runs measure. Clamped to [thresar_min, thresar_max].
        """
        if self.thresar_min == self.thresar_max:
            return self.thresar
        p0 = self.thresar
        nb1 = min(int(nb), 50)
        coeff = []
        for row in self._AR_POLY_COEFFS:
            c = row[0]
            for kj in range(1, 5):
                c = c * p0 + row[kj]
            coeff.append(c)
        th = coeff[0]
        for ki in range(1, 3):
            th = th / (nb1 + 1) + coeff[ki]
        return min(max(th, self.thresar_min), self.thresar_max)

    def double_difference(self, keys, elevations):
        """Pick a reference per (constellation, frequency) and pair the rest.

        ``keys`` are (sat, freq) tuples; ``elevations`` maps sat -> radians.
        Returns the list of (ref_key, target_key) pairs, reference first.
        """
        by_group = {}
        for sat, freq in keys:
            if elevations.get(int(sat), -np.inf) < self.el_mask:
                continue
            sys_i, _ = sat2prn(int(sat))
            by_group.setdefault((sys_i, int(freq)), []).append((int(sat), int(freq)))
        pairs = []
        for group in sorted(by_group):
            members = sorted(by_group[group])
            if len(members) < 2:
                continue
            ref = members[0]
            for key in members[1:]:
                pairs.append((ref, key))
        return pairs

    def resolve(self, float_values, covariance, keys, elevations):
        """Fix the ambiguities.

        Args:
          float_values: (sat, freq) -> float ambiguity [cycles].
          covariance:   joint covariance of ``keys`` in that order [cycles^2].
          keys:         ordered (sat, freq) tuples matching ``covariance``.
          elevations:   sat -> elevation [rad], for the reference choice.

        Returns an :class:`ResolverResult`; ``nb == 0`` means no fix was accepted.
        """
        keys = list(keys)
        pairs = self.double_difference(keys, elevations)
        if len(pairs) < self.min_pairs:
            return ResolverResult(pairs=pairs)

        index = {k: i for i, k in enumerate(keys)}
        # D maps the single-difference states onto the double differences.
        D = np.zeros((len(pairs), len(keys)))
        for row, (ref, tgt) in enumerate(pairs):
            D[row, index[ref]] = 1.0
            D[row, index[tgt]] = -1.0

        x = np.array([float_values[k] for k in keys], dtype=float)
        y = D @ x
        Q = D @ np.asarray(covariance, dtype=float) @ D.T

        b, s, nfix, ps = mlambda(y, Q, parmode=self.parmode, P0=self.par_p0)
        s0 = float(s[0]) if len(s) > 0 else 0.0
        s1 = float(s[1]) if len(s) > 1 else 0.0
        if s0 <= 1e-12 * max(s1, 1.0):
            # Exact fit (held integers re-entering): s0 is FP noise
            # and the ratio measures nothing.
            s0 = 0.0
        ratio = 0.0 if s0 <= 0.0 else s1 / s0
        thres = self.ratio_threshold(len(pairs))
        result = ResolverResult(s0=s0, s1=s1, pairs=pairs, thres_used=thres)
        if nfix <= 0:
            return result
        # s0 <= 0 means mlambda could not form a ratio; partial AR (parmode 2)
        # carries its own acceptance, so neither case is held to the threshold.
        if not (self.parmode == 2 or s0 <= 0.0 or ratio >= thres):
            return result

        # Restore single-difference ambiguities: the reference keeps its float
        # value and each target follows from the fixed difference.
        fixed = {}
        for row, (ref, tgt) in enumerate(pairs):
            fixed.setdefault(ref, float_values[ref])
            fixed[tgt] = fixed[ref] - float(b[row, 0])
        if self.parmode == 2 and int(nfix) < len(pairs):
            # Partial back-substitution is not implemented: decline
            # rather than hold non-fixed ambiguities at made-up integers.
            result.declined_partial = True
            return result
        result.nb = len(pairs)
        result.fixed = fixed
        return result
