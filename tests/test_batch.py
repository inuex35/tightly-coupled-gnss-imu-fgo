"""gnss_fgo.batch: union-find relations, chain selected inversion, LAMBDA."""
import numpy as np
import pytest
import scipy.sparse as sp

from gnss_fgo.batch import lambda_c
from gnss_fgo.batch.ar import ArcUnion
from gnss_fgo.batch.selinv import selected_inverse


def test_arc_union_transitive_and_conflict():
    uf = ArcUnion()
    assert uf.add(0, 1, 5) == "new"        # N1 - N0 = 5
    assert uf.add(1, 2, -3) == "new"       # N2 - N1 = -3
    assert uf.add(0, 2, 2) == "same"       # implied: N2 - N0 = 2
    assert uf.add(2, 0, -2) == "same"
    assert uf.add(0, 2, 3) == "conflict"
    assert uf.resolved(0, 2) and not uf.resolved(0, 7)
    r0, o0 = uf.find(0)
    r2, o2 = uf.find(2)
    assert r0 == r2 and o2 - o0 == 2


def _chain_problem(rng, n_ep=12, sdim=2, arcs=((0, 5), (2, 9), (4, 11), (0, 11), (7, 10))):
    """Random SPD information over epoch states (sdim each, linked k-1 -> k)
    and arcs alive over [first, last]; every factor touches variables that
    coexist at one epoch, as in the batch graphs."""
    na = len(arcs)
    n = na + sdim * n_ep
    xcol = lambda k: na + sdim * k  # noqa: E731
    H = np.zeros((n, n))

    def add(cols):
        J = rng.normal(size=(len(cols) + 1, len(cols)))
        H[np.ix_(cols, cols)] += J.T @ J

    for k in range(n_ep):
        alive = [a for a, (f, l) in enumerate(arcs) if f <= k <= l]
        xs = list(range(xcol(k), xcol(k) + sdim))
        add(xs + alive)
        if k > 0:
            add(list(range(xcol(k - 1), xcol(k - 1) + sdim)) + xs)
    for a in range(na):
        H[a, a] += 1e-2                       # weak gauge prior
    ent = np.zeros(n, np.int64)
    eli = np.zeros(n, np.int64)
    for a, (f, l) in enumerate(arcs):
        ent[a], eli[a] = f, l
    for k in range(n_ep):
        ent[xcol(k):xcol(k) + sdim] = max(k - 1, 0)
        eli[xcol(k):xcol(k) + sdim] = k
    epoch_cols = {k: [a for a, (f, l) in enumerate(arcs) if f <= k <= l] for k in range(n_ep)}
    return H, ent, eli, n_ep, epoch_cols


def test_selected_inverse_matches_dense_inverse():
    H, ent, eli, n_ep, epoch_cols = _chain_problem(np.random.default_rng(0))
    Q = selected_inverse(sp.coo_matrix(H), ent, eli, n_ep, epoch_cols)
    S = np.linalg.inv(H)
    for k, cols in epoch_cols.items():
        np.testing.assert_allclose(Q[k], S[np.ix_(cols, cols)], rtol=1e-8, atol=1e-10)


def _lambda_case():
    # Classic 3-D example (Teunissen); integer LS solution is (5, 3, 4).
    Q = np.array([[6.2900, 5.9780, 0.5440], [5.9780, 6.2920, 2.3400], [0.5440, 2.3400, 6.2880]])
    return np.array([5.45, 3.10, 2.97]), Q


def test_lambda_python_known_answer():
    a, Q = _lambda_case()
    best, _, s, info = lambda_c.lambda_batch([a], [Q], backend="python")
    assert list(best[0]) == [5, 3, 4]
    assert s[0, 1] >= s[0, 0] and info[0] == 0


def test_lambda_c_matches_python():
    if not (lambda_c.available() or lambda_c.build()):
        pytest.skip("no C compiler for lambda_c")
    rng = np.random.default_rng(1)
    ahats, Qs = [], []
    for n in (3, 6, 10, 17):
        A = rng.normal(size=(n, n))
        Qs.append(A @ A.T * 0.05 + np.eye(n) * 0.01)
        ahats.append(rng.normal(size=n) * 20)
    a, Q = _lambda_case()
    ahats.append(a)
    Qs.append(Q)
    bc, _, sc, ic = lambda_c.lambda_batch(ahats, Qs, backend="c")
    bp, _, sp_, _ = lambda_c.lambda_batch(ahats, Qs, backend="python")
    assert np.all(ic == 0)
    for x, y in zip(bc, bp):
        assert list(x) == list(y)
    np.testing.assert_allclose(sc, sp_, rtol=1e-8)
