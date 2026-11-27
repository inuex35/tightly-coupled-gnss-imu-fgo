"""Batch ambiguity resolution: selected inversion (Takahashi), ArcUnion, and LAMBDA."""

from concurrent.futures import ProcessPoolExecutor
import os

import gtsam
import numpy as np
import scipy.sparse as sp
from cssrlib.core.mlambda import mlambda
from gtsam.symbol_shorthand import N
from scipy.linalg import cho_factor, cho_solve

from .ar import AmbiguityResolver, _FFRT_MAX_RATIO, _FFRT_MIN_RATIO
from .batch_graph import ARC_REF, ARC_TGT, EPOCH, SATNO_TGT

_MIN_PARALLEL = 256     # below this, pool start-up costs more than it saves
_FFRT = AmbiguityResolver(
    thresar=0.001, thresar_min=_FFRT_MIN_RATIO, thresar_max=_FFRT_MAX_RATIO)


def _solve_one(a, Q):
    try:
        afix, s, _, _ = mlambda(np.asarray(a, float), np.asarray(Q, float), ncands=2)
    except (np.linalg.LinAlgError, ValueError):
        n = len(a)
        return np.zeros(n, int), np.zeros(n, int), np.zeros(2), -1
    return (np.rint(afix[:, 0]).astype(int), np.rint(afix[:, 1]).astype(int),
            np.asarray(s[:2], float), 0)


def _solve_chunk(chunk):
    return [_solve_one(a, Q) for a, Q in chunk]


def lambda_batch(ahats, Qs, workers=0):
    """Integer LS for each (ahat, Q).

    Returns (best, second, s, info): best/second integer vectors per problem,
    s (nprob, 2) squared residuals (ratio = s[:, 1] / s[:, 0]), info 0 = ok.
    workers: processes to use; 0 = os.cpu_count(), 1 = serial.
    """
    probs = list(zip(ahats, Qs))
    workers = workers or os.cpu_count() or 1
    if workers <= 1 or len(probs) < _MIN_PARALLEL:
        out = _solve_chunk(probs)
    else:
        size = -(-len(probs) // (4 * workers))
        chunks = [probs[i:i + size] for i in range(0, len(probs), size)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = [r for part in ex.map(_solve_chunk, chunks) for r in part]
    best = [o[0] for o in out]
    second = [o[1] for o in out]
    s = np.array([o[2] for o in out], dtype=float).reshape(-1, 2)
    info = np.array([o[3] for o in out], dtype=np.int32)
    return best, second, s, info


# ── Selected inversion (Takahashi) along the time chain ───────────────

def information_matrix(graph, values, dims):
    """H = A^T A of graph.linearize(values) and {key: first column}."""
    col, c = {}, 0
    for key in sorted(graph.keys()):
        col[key] = c
        c += dims[chr(gtsam.Symbol(key).chr())]
    ijs = graph.linearize(values).sparseJacobian_()
    i, j, s = ijs[0].astype(np.int64) - 1, ijs[1].astype(np.int64) - 1, ijs[2]
    keep = j < c
    A = sp.csr_matrix((s[keep], (i[keep], j[keep])), shape=(int(i.max()) + 1, c))
    return (A.T @ A).tocoo(), col


def selected_inverse(H, ent, eli, n_ep, epoch_cols):
    """Joint covariance of epoch_cols[k] (column ids, in order) for each k."""
    t = np.maximum(ent[H.row], ent[H.col])
    order = np.argsort(t, kind="stable")
    rows, cols, vals, t = H.row[order], H.col[order], H.data[order], t[order]
    bounds = np.searchsorted(t, np.arange(n_ep + 1))
    by_ent = np.argsort(ent, kind="stable")
    enter_at = np.split(by_ent, np.searchsorted(ent[by_ent], np.arange(1, n_ep)))

    front, pos, F = [], {}, np.zeros((0, 0))
    steps = []
    for k in range(n_ep):
        new = enter_at[k]
        if len(new):
            m0 = len(front)
            front.extend(new.tolist())
            for q, c in enumerate(new.tolist()):
                pos[c] = m0 + q
            G = np.zeros((len(front), len(front)))
            G[:m0, :m0] = F
            F = G
        lo, hi = bounds[k], bounds[k + 1]
        if hi > lo:
            pi = np.fromiter((pos[c] for c in rows[lo:hi]), np.int64, hi - lo)
            pj = np.fromiter((pos[c] for c in cols[lo:hi]), np.int64, hi - lo)
            np.add.at(F, (pi, pj), vals[lo:hi])
        e_mask = np.array([eli[c] == k for c in front])
        ei, ri = np.nonzero(e_mask)[0], np.nonzero(~e_mask)[0]
        Acho = cho_factor(F[np.ix_(ei, ei)], lower=True)
        B = F[np.ix_(ri, ei)]
        E_cols, R_cols = [front[q] for q in ei], [front[q] for q in ri]
        steps.append((E_cols, R_cols, Acho, B))
        F = F[np.ix_(ri, ri)] - B @ cho_solve(Acho, B.T)
        front = list(R_cols)
        pos = {c: q for q, c in enumerate(front)}

    out, sig_cols, Sig = {}, [], np.zeros((0, 0))
    for k in range(n_ep - 1, -1, -1):
        E_cols, R_cols, Acho, B = steps[k]
        if R_cols:
            spos = {c: q for q, c in enumerate(sig_cols)}
            idx = np.array([spos[c] for c in R_cols])
            S_RR = Sig[np.ix_(idx, idx)]
            W = cho_solve(Acho, B.T)
            S_ER = -W @ S_RR
            S_EE = cho_solve(Acho, np.eye(len(E_cols))) - S_ER @ W.T
            Sig = np.block([[S_EE, S_ER], [S_ER.T, S_RR]])
        else:
            Sig = cho_solve(Acho, np.eye(len(E_cols)))
        sig_cols = E_cols + R_cols
        if k in epoch_cols:
            spos = {c: q for q, c in enumerate(sig_cols)}
            idx = np.array([spos[c] for c in epoch_cols[k]])
            out[k] = Sig[np.ix_(idx, idx)]
    return out


def chain_columns(col, dims, n_ep, arc_first, arc_last):
    """ent / eli per column for keys x/v/b(k), n(a), and global symbols."""
    n = max(c0 + dims[chr(gtsam.Symbol(key).chr())] for key, c0 in col.items())
    ent = np.zeros(n, dtype=np.int64)
    eli = np.zeros(n, dtype=np.int64)
    for key, c0 in col.items():
        sym = gtsam.Symbol(key)
        ch, idx = chr(sym.chr()), sym.index()
        d = dims[ch]
        if ch == "n":
            ent[c0], eli[c0] = arc_first.get(idx, 0), arc_last.get(idx, 0)
        elif ch in "xvb":
            ent[c0:c0 + d], eli[c0:c0 + d] = max(idx - 1, 0), idx
        else:
            ent[c0:c0 + d], eli[c0:c0 + d] = 0, n_ep - 1
    return ent, eli


# ── ArcUnion & per-epoch LAMBDA ───────────────────────────────────────

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
    """{k: (pairs, arcs)} from epoch-sorted DD rows."""
    ep = rows[:, EPOCH].astype(int)
    if np.any(np.diff(ep) < 0):
        raise ValueError("DD rows must be sorted by epoch")
    ks, start = np.unique(ep, return_index=True)
    out = {}
    for k, r in zip(ks, np.split(rows[:, [ARC_REF, ARC_TGT]].astype(int), start[1:])):
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


def _solve_partial_one(a, Q, min_pairs=4):
    """Full LAMBDA, falling back to Partial AR with dimension-adaptive FFRT."""
    keep = np.arange(len(a))
    while len(keep) >= min_pairs:
        best, _, s, info = _solve_one(a[keep], Q[np.ix_(keep, keep)])
        if info == 0:
            ratio = float(s[1] / max(s[0], 1e-12))
            if ratio >= _FFRT.ratio_threshold(len(keep)):
                return keep, best, ratio
        if len(keep) == min_pairs:
            break
        drop = int(np.argmax(np.diag(Q[np.ix_(keep, keep)])))
        keep = np.delete(keep, drop)
    return None, None, 0.0


def _solve_partial_chunk(chunk):
    return [_solve_partial_one(a, Q, min_pairs) for a, Q, min_pairs in chunk]


def resolve(groups, values, cov, ratio_min=None, workers=0, min_pairs=4):
    """Returns (accepted relations [(r, t, n)], sources [(epoch, ratio)], passed, stats)."""
    probs = dd_problems(groups, values, cov)
    stats = {"problems": len(probs), "new": 0, "same": 0, "conflict": 0}
    if not probs:
        return [], [], set(), stats
    best, _, s, info = lambda_batch([p[2] for p in probs], [p[3] for p in probs], workers)
    ratio = np.where(info == 0, s[:, 1] / np.maximum(s[:, 0], 1e-12), 0.0)

    cand = []
    retry = []
    for i, p in enumerate(probs):
        nb = len(p[1])
        thresh = ratio_min if ratio_min is not None else _FFRT.ratio_threshold(nb)
        if ratio[i] >= thresh:
            cand.append((float(ratio[i]), p[0], p[1], best[i]))
        elif nb > min_pairs:
            retry.append((i, p[2], p[3]))

    if retry:
        tasks = [(a, Q, min_pairs) for _, a, Q in retry]
        w = workers or os.cpu_count() or 1
        if w <= 1 or len(tasks) < _MIN_PARALLEL:
            par_out = _solve_partial_chunk(tasks)
        else:
            size = max(1, -(-len(tasks) // (4 * w)))
            chunks = [tasks[j:j + size] for j in range(0, len(tasks), size)]
            with ProcessPoolExecutor(max_workers=w) as ex:
                par_out = [r for part in ex.map(_solve_partial_chunk, chunks) for r in part]
        for (i, _, _), (keep, p_best, p_rat) in zip(retry, par_out):
            if keep is not None:
                p = probs[i]
                cand.append((p_rat, p[0], p[1][keep], p_best))

    cand.sort(key=lambda c: -c[0])
    uf, accepted, sources, passed = ArcUnion(), [], [], set()
    for rat, k, pairs, ints in cand:
        trial = ArcUnion()
        trial.parent = uf.parent.copy()
        trial.off = uf.off.copy()
        res = [trial.add(int(r), int(t), int(n)) for (r, t), n in zip(pairs, ints)]
        if "conflict" in res:
            stats["conflict"] += res.count("conflict")
            continue
        uf = trial
        for (r, t), n, st in zip(pairs, ints, res):
            stats[st] += 1
            if st == "new":
                accepted.append((int(r), int(t), int(n)))
                sources.append((k, float(rat)))
        passed.add(k)
    stats["ratio_pass"] = len(cand)
    return accepted, sources, passed, stats


def validate_fix(rows, keep_cp, accepted, cp_res, n_ep, cfg):
    """Per-epoch FIX validation of the fixed solution."""
    uf = ArcUnion(accepted)
    k = rows[:, EPOCH].astype(int)
    resolved = np.array([keep_cp[i] and uf.resolved(rows[i, ARC_REF], rows[i, ARC_TGT])
                         for i in range(len(rows))], dtype=bool)
    n_res = np.bincount(k[resolved], minlength=n_ep)
    worst = np.zeros(n_ep)
    np.maximum.at(worst, k[resolved], np.abs(np.nan_to_num(cp_res[resolved], nan=np.inf)))
    sats = np.zeros(n_ep, dtype=int)
    for kk, _ in set(zip(k[resolved], rows[resolved, SATNO_TGT].astype(int))):
        sats[kk] += 1
    enough = (n_res >= cfg.fix_min_pairs) & (sats >= cfg.fix_min_sats)
    fix = enough & (worst <= cfg.fix_res)
    return fix, enough & ~fix, worst
