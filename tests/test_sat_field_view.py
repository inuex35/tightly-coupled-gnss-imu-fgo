"""CurrentEpochState lifetime contract."""
from gnss_fgo.state.runtime_state import CurrentEpochState


def test_current_epoch_lifetime_is_object_replacement():
    scratch = CurrentEpochState()
    scratch.ref_sats[1] = 7
    scratch.seeded_amb_keys.add((7, 0))
    fresh = CurrentEpochState()          # what _reset_current_epoch does
    assert fresh.ref_sats == {} and fresh.seeded_amb_keys == set()
    # the old epoch's state is unreachable, not "wiped in place"
    assert scratch.ref_sats == {1: 7}
