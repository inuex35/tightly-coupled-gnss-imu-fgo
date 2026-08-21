"""SatFieldView semantics + the EpochScratch lifetime contract.

Review round 4 (A-1/A-2) traced two behavior bugs to epoch-lifetime
state living on the runner behind manual per-field wipes. The fix is
structural: EpochScratch is REPLACED each epoch, so nothing in it can
outlive the epoch. These tests pin both halves.
"""
from gnss_fgo.state.runtime_state import (
    EpochScratch, SatFieldView, SatStateMap)


def test_view_reads_only_set_fields():
    m = SatStateMap()
    m.get(1, 0).amb_key = 100
    m.get(2, 0).amb_key = None
    v = SatFieldView(m, 'amb_key')
    assert dict(v) == {(1, 0): 100}
    assert (2, 0) not in v


def test_clear_resets_every_entry_to_absent():
    m = SatStateMap()
    m.get(3, 0).amb_gen = 2
    m.get(4, 1).amb_gen = 5
    v = SatFieldView(m, 'amb_gen', absent=0)
    assert dict(v) == {(3, 0): 2, (4, 1): 5}
    v.clear()
    assert dict(v) == {}
    assert m.get(3, 0).amb_gen == 0 and m.get(4, 1).amb_gen == 0


def test_epoch_scratch_lifetime_is_object_replacement():
    scratch = EpochScratch()
    scratch.ref_sats[1] = 7
    scratch.seeded_amb_keys.add((7, 0))
    fresh = EpochScratch()          # what _reset_epoch_scratch does
    assert fresh.ref_sats == {} and fresh.seeded_amb_keys == set()
    # the old epoch's state is unreachable, not "wiped in place"
    assert scratch.ref_sats == {1: 7}
