"""The Phase-2 stage-declared data-flow contract must hold.

Every pipeline stage declares STAGE_READS / STAGE_WRITES; this test
walks the stages in pipeline order and fails if any stage reads an
EpochData field that no earlier stage (or the dataclass init) provides,
or touches a field that does not exist. Runtime code only checks this
when ENABLE_STAGE_CONTRACT_CHECK=1; the test makes it unconditional
in CI.
"""
from gnss_fgo.state.stage_contract import validate_pipeline


def test_stage_contract_holds():
    errors, summary = validate_pipeline()
    assert not errors, '\n'.join(errors)
    # All five stages must actually declare their I/O — an empty
    # declaration would make the walk pass vacuously.
    for name, reads, writes in summary:
        assert reads or writes, f'stage {name!r} declares no I/O'
