"""Selected inversion (Takahashi) along the time chain of a batch graph.

The information matrix H = A^T A of the linearized graph is factored with a
time-ordered block elimination: at step k every column whose last epoch is k
(the epoch's states, and the SD ambiguity arcs ending at k) is eliminated from
a small dense front holding only what is still alive. A backward sweep (the
RTS-smoother form of the Takahashi recursion) then yields the joint covariance
of each front, which contains every arc observed at epoch k. One forward and
one backward pass replace thousands of Bayes-tree joint-marginal queries, and
plain Cholesky is stable in this ordering where Marginals(Cholesky) is not.
"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve


def information_matrix(graph, values, dims):
    """H = A^T A of graph.linearize(values) and {key: first column}.

    Columns follow GaussianFactorGraph.sparseJacobian_(): keys in sorted
    order. dims: {symbol character: tangent dimension}, e.g. {'x': 6, 'n': 1}.
    """
    import gtsam
    col, c = {}, 0
    for key in sorted(graph.keys()):
        col[key] = c
        c += dims[chr(gtsam.Symbol(key).chr())]
    ijs = graph.linearize(values).sparseJacobian_()
    i, j, s = ijs[0].astype(np.int64) - 1, ijs[1].astype(np.int64) - 1, ijs[2]
    keep = j < c                       # drop the RHS column
    A = sp.csr_matrix((s[keep], (i[keep], j[keep])), shape=(int(i.max()) + 1, c))
    return (A.T @ A).tocoo(), col


def selected_inverse(H, ent, eli, n_ep, epoch_cols):
    """Joint covariance of epoch_cols[k] (column ids, in order) for each k.

    ent[c] / eli[c]: step at which column c enters the front / is eliminated.
    Every nonzero H[i, j] must have max(ent) <= min(eli) over its two
    columns, which holds when each factor only touches variables that are
    alive together at some step.
    """
    t = np.maximum(ent[H.row], ent[H.col])   # add each entry when both columns exist
    order = np.argsort(t, kind="stable")
    rows, cols, vals, t = H.row[order], H.col[order], H.data[order], t[order]
    bounds = np.searchsorted(t, np.arange(n_ep + 1))
    by_ent = np.argsort(ent, kind="stable")
    enter_at = np.split(by_ent, np.searchsorted(ent[by_ent], np.arange(1, n_ep)))

    front, pos, F = [], {}, np.zeros((0, 0))
    steps = []                  # (E cols, R cols, cho(A), B) per step
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
        front = list(R_cols)    # copy: R_cols is kept in steps
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
    """ent / eli per column for keys x/v/b(k) (epoch states), n(a) (arcs) and
    any other symbol (global: alive for the whole run)."""
    import gtsam
    n = max(c0 + dims[chr(gtsam.Symbol(key).chr())] for key, c0 in col.items())
    ent = np.zeros(n, dtype=np.int64)
    eli = np.zeros(n, dtype=np.int64)
    for key, c0 in col.items():
        sym = gtsam.Symbol(key)
        ch, idx = chr(sym.chr()), sym.index()
        d = dims[ch]
        if ch == "n":   # an arc whose CP rows were all excluded keeps only its prior
            ent[c0], eli[c0] = arc_first.get(idx, 0), arc_last.get(idx, 0)
        elif ch in "xvb":
            ent[c0:c0 + d], eli[c0:c0 + d] = max(idx - 1, 0), idx
        else:
            ent[c0:c0 + d], eli[c0:c0 + d] = 0, n_ep - 1
    return ent, eli
