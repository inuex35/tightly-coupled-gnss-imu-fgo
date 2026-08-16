"""Integer ambiguity resolution over the graph's own float estimate.

The pipeline reaches LAMBDA through cssrlib's EKF class: it writes the
smoother's marginals into ``nav.x`` / ``nav.P``, sets ``nav.vsat`` / ``nav.fix``
and calls ``resamb_lambda``, which reads them back out. The EKF itself never
runs -- ``kfupdate`` and ``udstate`` are not called anywhere in Phase 2 -- so
what the inheritance buys is four functions (``ddidx``, ``mlambda``,
``restamb``, the ratio test) at the price of a state container that has to be
kept in sync by hand, and of contracts that are invisible until they break
(the ``_last_s0`` ratio stash was silently dropped in a cssrlib refactor and
cost 0.85 m 3D RMS on tokyo run2 before anyone noticed).

This resolver takes what the graph already has -- float ambiguities and their
joint covariance -- and returns the fixed ones. No shared state, no index
arithmetic against ``nav.na``, and the ratio comes back with the result
instead of being left on the engine for the caller to find.

The double-difference construction matches cssrlib's ``ddidx``: within each
constellation and frequency, the satellite with the highest elevation is the
reference and every other satellite is differenced against it.
"""

from dataclasses import dataclass, field

import numpy as np
from cssrlib.gnss import sat2prn
from cssrlib.mlambda import mlambda


@dataclass
class ArResult:
    """Outcome of one resolution attempt."""

    nb: int = 0                       # number of fixed DD ambiguities
    fixed: dict = field(default_factory=dict)   # (sat, freq) -> fixed SD value
    ratio: float = 0.0                # s[1]/s[0], 0 when unavailable
    s0: float = 0.0
    s1: float = 0.0
    nfix: int = 0                     # integer-fixed count reported by mlambda
    ps: float = 0.0                   # success rate from partial AR
    pairs: list = field(default_factory=list)   # (ref, target, freq) per DD


class AmbiguityResolver:
    """LAMBDA over a float ambiguity vector and its covariance."""

    def __init__(self, thresar=3.0, parmode=1, par_p0=0.995, min_pairs=2,
                 el_mask=0.0):
        self.thresar = float(thresar)
        self.parmode = int(parmode)
        self.par_p0 = float(par_p0)
        self.min_pairs = int(min_pairs)
        self.el_mask = float(el_mask)

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

        Returns an :class:`ArResult`; ``nb == 0`` means no fix was accepted.
        """
        keys = list(keys)
        pairs = self.double_difference(keys, elevations)
        if len(pairs) < self.min_pairs:
            return ArResult(pairs=pairs)

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
        ratio = 0.0 if s0 <= 0.0 else s1 / s0
        result = ArResult(ratio=ratio, s0=s0, s1=s1, nfix=int(nfix),
                          ps=float(ps), pairs=pairs)
        if nfix <= 0:
            return result
        # s0 <= 0 means mlambda could not form a ratio; partial AR (parmode 2)
        # carries its own acceptance, so neither case is held to the threshold.
        if not (self.parmode == 2 or s0 <= 0.0 or ratio >= self.thresar):
            return result

        # Restore single-difference ambiguities: the reference keeps its float
        # value and each target follows from the fixed difference.
        fixed = {}
        for row, (ref, tgt) in enumerate(pairs):
            fixed.setdefault(ref, float_values[ref])
            fixed[tgt] = fixed[ref] - float(b[row, 0])
        result.nb = len(pairs)
        result.fixed = fixed
        return result
