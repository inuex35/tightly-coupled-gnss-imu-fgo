"""Batch ambiguity resolution: per-epoch LAMBDA, merged per SD arc.

1. Joint covariance of the SD arcs in use at every epoch from one time-chain
   selected inversion of the float solution (selinv).
2. DD float ambiguities N_tgt - N_ref and their covariance per epoch ->
   batched LAMBDA + ratio test.
3. Accepted DD integers are relations between SD arcs; a union-find with
   integer offsets merges them across epochs, strongest ratio first, and
   rejects relations that contradict the ones already merged. A relation
   then constrains its arcs over their whole life (both directions in time,
   and transitively across satellites) -- the batch counterpart of
   fix-and-hold.
"""
import numpy as np
from gtsam.symbol_shorthand import N

from . import rows as R
from .lambda_c import lambda_batch
from .selinv import chain_columns, information_matrix, selected_inverse


class ArcUnion:
    """Union-find over SD arcs keeping integer offsets: N[a] - N[root] = off[a]."""

    def __init__(self, relations=()):
        self.parent, self.off = {}, {}
        for r, t, n in relations:
            self.add(int(r), int(t), int(n))

    def find(self, a):
        if a not in self.parent:
            self.parent[a], self.off[a] = a, 0
            return a, 0
        if self.parent[a] == a:
            return a, 0
        root, o = self.find(self.parent[a])
        self.parent[a], self.off[a] = root, self.off[a] + o
        return root, self.off[a]

    def add(self, r, t, n):
        """Relation N[t] - N[r] = n. Returns 'new', 'same' or 'conflict'."""
        rr, orr = self.find(r)
        rt, ort = self.find(t)
        if rr == rt:
            return "same" if ort - orr == n else "conflict"
        self.parent[rt], self.off[rt] = rr, orr + n - ort
        return "new"

    def resolved(self, r, t):
        return self.find(int(r))[0] == self.find(int(t))[0]


def group_epochs(rows):
    """{k: (pairs, arcs)} from epoch-sorted DD rows; pairs are unique
    (ref arc, tgt arc), arcs the sorted arcs used (the covariance order)."""
    ep = rows[:, R.EPOCH].astype(int)
    if np.any(np.diff(ep) < 0):
        raise ValueError("DD rows must be sorted by epoch")
    ks, start = np.unique(ep, return_index=True)
    out = {}
    for k, r in zip(ks, np.split(rows[:, [R.ARC_REF, R.ARC_TGT]].astype(int), start[1:])):
        pairs = np.unique(r, axis=0)
        out[int(k)] = (pairs, np.unique(pairs))
    return out


def arc_covariances(graph, values, groups, n_ep, dims):
    """{k: joint covariance of groups[k] arcs} via selected inversion."""
    H, col = information_matrix(graph, values, dims)
    first, last = {}, {}
    for k, (_, arcs) in groups.items():
        for a in arcs:
            first.setdefault(int(a), k)
            last[int(a)] = k
    ent, eli = chain_columns(col, dims, n_ep, first, last)
    epoch_cols = {k: [col[N(int(a))] for a in arcs] for k, (_, arcs) in groups.items()}
    return selected_inverse(H, ent, eli, n_ep, epoch_cols)


def dd_problems(groups, values, cov):
    """[(k, pairs, ahat, Qa)]: DD float ambiguities and covariance per epoch."""
    narc = max(int(arcs[-1]) for _, arcs in groups.values()) + 1
    sd_all = np.array([values.atDouble(N(a)) if values.exists(N(a)) else np.nan
                       for a in range(narc)])
    probs = []
    for k, (pairs, arcs) in groups.items():
        if len(pairs) < 4:
            continue
        ir = np.searchsorted(arcs, pairs[:, 0])
        it = np.searchsorted(arcs, pairs[:, 1])
        Q = cov[k]
        sd = sd_all[arcs]
        Qa = Q[np.ix_(it, it)] - Q[np.ix_(it, ir)] - Q[np.ix_(ir, it)] + Q[np.ix_(ir, ir)]
        probs.append((k, pairs, sd[it] - sd[ir], Qa))
    return probs


def resolve(groups, values, cov, ratio_min=3.0, backend="auto"):
    """Returns (accepted relations [(r, t, n)], sources [(epoch, ratio)],
    ratio-passing epochs, stats dict)."""
    probs = dd_problems(groups, values, cov)
    stats = {"problems": len(probs), "new": 0, "same": 0, "conflict": 0}
    if not probs:
        return [], [], set(), stats
    best, _, s, info = lambda_batch([p[2] for p in probs], [p[3] for p in probs], backend)
    ratio = np.where(info == 0, s[:, 1] / np.maximum(s[:, 0], 1e-12), 0.0)
    cand = [(ratio[i], p[0], p[1], best[i]) for i, p in enumerate(probs)
            if ratio[i] >= ratio_min]
    cand.sort(key=lambda c: -c[0])   # strongest first: a weak wrong fix cannot block it
    uf, accepted, sources, passed = ArcUnion(), [], [], set()
    for rat, k, pairs, ints in cand:
        res = [uf.add(int(r), int(t), int(n)) for (r, t), n in zip(pairs, ints)]
        for (r, t), n, st in zip(pairs, ints, res):
            stats[st] += 1
            if st == "new":
                accepted.append((int(r), int(t), int(n)))
                sources.append((k, float(rat)))
        if "conflict" not in res:
            passed.add(k)
    stats["ratio_pass"] = len(cand)
    return accepted, sources, passed, stats


def validate_fix(rows, keep_cp, accepted, cp_res, n_ep, cfg):
    """Per-epoch FIX of the fixed solution: >= fix_min_pairs carrier DD pairs
    and >= fix_min_sats distinct target satellites with a resolved integer,
    all fitting within fix_res metres. Returns (fix, resolved-but-rejected,
    worst |residual| per epoch)."""
    uf = ArcUnion(accepted)
    k = rows[:, R.EPOCH].astype(int)
    resolved = np.array([keep_cp[i] and uf.resolved(rows[i, R.ARC_REF], rows[i, R.ARC_TGT])
                         for i in range(len(rows))], dtype=bool)
    n_res = np.bincount(k[resolved], minlength=n_ep)
    worst = np.zeros(n_ep)
    np.maximum.at(worst, k[resolved], np.abs(np.nan_to_num(cp_res[resolved], nan=np.inf)))
    sats = np.zeros(n_ep, dtype=int)
    for kk, _ in set(zip(k[resolved], rows[resolved, R.SATNO_TGT].astype(int))):
        sats[kk] += 1
    enough = (n_res >= cfg.fix_min_pairs) & (sats >= cfg.fix_min_sats)
    fix = enough & (worst <= cfg.fix_res)
    return fix, enough & ~fix, worst
