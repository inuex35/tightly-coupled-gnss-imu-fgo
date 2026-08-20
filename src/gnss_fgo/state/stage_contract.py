"""Phase-2 pipeline data-flow contract checker.

Each stage module under ``pipeline/`` declares two module-level tuples:

  ``STAGE_READS``   — fields the stage reads from the shared EpochData.
  ``STAGE_WRITES``  — fields the stage writes (or in-place mutates) on it.

This module loads the five stages in pipeline order and verifies, at
import time, that no stage reads a field that is neither (a) populated
by EpochData's defaults / dataclass init, nor (b) written by an
earlier stage.

The check is opt-in via ``ENABLE_STAGE_CONTRACT_CHECK=1``; it is meant
as a sanity guardrail during development and CI, not a runtime cost.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import MISSING

from .epoch_data import EpochData


def _seed_fields():
    """Fields that are populated before any stage runs.

    Includes (a) required fields the caller must pass at construction
    (no default) and (b) ``field(default_factory=...)`` fields whose
    factory yields a meaningful container. Excludes sentinel defaults
    like ``None`` / ``0`` / ``False`` that explicitly mark "must be
    written by some stage before any other stage may read".
    """
    out = set()
    for name, f in EpochData.__dataclass_fields__.items():
        if f.default is MISSING and f.default_factory is MISSING:
            out.add(name)        # required positional
        elif f.default_factory is not MISSING:
            out.add(name)        # factory-produced container
    return out


# (stage-name, dotted module path under gnss_fgo) — order matches the
# Phase-2 pipeline data flow.
_DEFAULT_ORDER = (
    ('A imu',     '.pipeline.imu_prediction'),
    ('B gate',    '.pipeline.quality_gate'),
    ('C solve',   '.pipeline.solve'),
    ('D verdict', '.pipeline.validate_fix'),
    ('E output',  '.pipeline.report'),
)


def _stage_modules(order=_DEFAULT_ORDER):
    return [(name, importlib.import_module(path, package='gnss_fgo'))
            for name, path in order]


def validate_pipeline(order=_DEFAULT_ORDER):
    """Walk the stage modules, return (errors, summary).

    Errors are returned as a list of strings; an empty list means the
    contract holds. ``summary`` is a list of (stage_name, reads, writes)
    tuples so callers can pretty-print the data-flow.
    """
    seeded = _seed_fields()
    all_fields = set(EpochData.__dataclass_fields__.keys())
    errors = []
    summary = []
    available = set(seeded)
    for name, mod in _stage_modules(order):
        reads = set(getattr(mod, 'STAGE_READS', ()) or ())
        writes_raw = set(getattr(mod, 'STAGE_WRITES', ()) or ())
        # Strip the `[*]` mutation marker for availability tracking.
        writes = {w.split('[')[0] for w in writes_raw}
        # A stage may read fields that it itself writes (intra-stage
        # data flow is fine); count its own writes as available too.
        available_in_stage = available | writes
        missing = reads - available_in_stage
        if missing:
            errors.append(
                f'stage {name!r} reads fields not yet provided: '
                f'{sorted(missing)}')
        unknown = (reads | writes) - all_fields
        if unknown:
            errors.append(
                f'stage {name!r} touches unknown epoch field(s): '
                f'{sorted(unknown)}')
        available |= writes
        summary.append((name, sorted(reads), sorted(writes_raw)))
    return errors, summary


def _maybe_check_at_import():
    if os.environ.get('ENABLE_STAGE_CONTRACT_CHECK', '0') != '1':
        return
    errors, _ = validate_pipeline()
    if errors:
        raise RuntimeError('Phase-2 pipeline contract violation:\n  '
                           + '\n  '.join(errors))


_maybe_check_at_import()
