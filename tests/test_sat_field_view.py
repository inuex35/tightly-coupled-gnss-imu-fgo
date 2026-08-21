"""SatFieldView semantics that A-1/A-2 depended on.

The per-epoch scratch reset assigns {} through these views, which must
clear the underlying per-satellite field back to its 'absent' value for
EVERY tracked satellite. Review round 4 (A-1) showed the whole
new-ambiguity AR gate rides on exactly this behavior: after the wipe,
a non-None amb_init_epoch can only mean 'seeded this epoch'.
"""
from gnss_fgo.state.runtime_state import SatFieldView, SatStateMap


def _map_with(entries):
    m = SatStateMap()
    for (s, f), ep in entries.items():
        m.get(s, f).amb_init_epoch = ep
    return m


def test_view_reads_only_set_fields():
    m = _map_with({(1, 0): 100, (2, 0): None})
    v = SatFieldView(m, 'amb_init_epoch')
    assert dict(v) == {(1, 0): 100}
    assert (2, 0) not in v


def test_clear_resets_every_entry_to_absent():
    m = _map_with({(1, 0): 100, (2, 1): 101})
    v = SatFieldView(m, 'amb_init_epoch')
    v.clear()
    assert dict(v) == {}
    # the A-1 invariant: after the wipe, non-None == seeded this epoch
    assert m.get(1, 0).amb_init_epoch is None
    assert m.get(2, 1).amb_init_epoch is None
    m.get(1, 0).amb_init_epoch = 200   # amb_seed writes during the epoch
    assert dict(v) == {(1, 0): 200}


def test_absent_sentinel_views_hide_the_sentinel():
    m = SatStateMap()
    m.get(3, 0).amb_gen = 0
    m.get(4, 0).amb_gen = 2
    v = SatFieldView(m, 'amb_gen', absent=0)
    assert dict(v) == {(4, 0): 2}
    v.clear()
    assert m.get(4, 0).amb_gen == 0
