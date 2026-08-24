"""Stage contracts must not over-declare: every declared READ/WRITE
field must actually be touched in the stage's source (its module plus
the pipeline sub-modules it imports, transitively). The forward check
catches missing supplies; this catches declarations nobody honours."""
import importlib
import importlib.util
import inspect
import re

from gnss_fgo.state.stage_contract import _DEFAULT_ORDER


def _stage_source(mod, seen=None):
    """Source of ``mod`` plus every gnss_fgo module it imports relatively."""
    seen = set() if seen is None else seen
    if mod.__name__ in seen:
        return ''
    seen.add(mod.__name__)
    src = inspect.getsource(mod)
    pkg = mod.__name__.rsplit('.', 1)[0]
    for m in re.finditer(r'^from (\.+)([\w.]*) import (.+)$', src, re.M):
        dots, base, names = m.groups()
        # `from .x import f` pulls module .x itself; `from . import x`
        # pulls module .x — try both shapes.
        cands = [('.' * len(dots) + base)] if base else []
        for name in names.split(','):
            name = name.strip().split(' as ')[0].strip()
            cands.append('.' * len(dots) + (base + '.' if base else '') + name)
        for rel in cands:
            try:
                sub = importlib.import_module(
                    importlib.util.resolve_name(rel, pkg))
            except (ImportError, ValueError):
                continue
            if inspect.ismodule(sub):
                src += _stage_source(sub, seen)
    return src


def _unhonoured(order=_DEFAULT_ORDER):
    out = []
    for name, path in order:
        mod = importlib.import_module(path, package='gnss_fgo')
        src = _stage_source(mod)
        for field in getattr(mod, 'STAGE_READS', ()):
            base = field.split('[')[0]
            if not re.search(rf'\bepoch\.{base}\b', src):
                out.append(f'{name} READS {field}')
        for field in getattr(mod, 'STAGE_WRITES', ()):
            base = field.split('[')[0]
            if '[' in field:      # in-place mutation of a container
                pat = (rf'\bepoch\.{base}\s*\[[^\]]*\]\s*=[^=]'
                       rf'|\bepoch\.{base}\.(append|update|add|clear|pop|extend|insert|discard)\(')
            else:
                pat = rf'\bepoch\.{base}\s*(=[^=]|,)'
            if not re.search(pat, src):
                out.append(f'{name} WRITES {field}')
    return out


def test_stage_contract_has_no_dead_declarations():
    dead = _unhonoured()
    assert not dead, '\n'.join(dead)


if __name__ == '__main__':
    print('\n'.join(_unhonoured()) or 'clean')
