"""Every info key the log reader consumes must have a producer.

The example's formatter once carried four branches for keys whose
producers had been deleted (DIRTY_RESET / DIRTY_PEN / UNTRUSTED_RES /
JUMP) — dead code that read as live features. This audit keeps reader
and producers honest. Regex-based: writes are `info['k'] =`,
dict-literal `'k': v` entries inside make_epoch_diagnostics, and
`info[reason + '_suffix']` dynamic writes (suffix-matched).
"""
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / 'src' / 'gnss_fgo'
EXAMPLE = ROOT / 'examples' / 'run_imu_gnss_tc.py'


def _collect():
    writes, dynamic_suffixes, reads = set(), set(), set()
    wpat = re.compile(r"info\[['\"](\w+)['\"]\]\s*=")
    dynpat = re.compile(r"info\[[\w. ]+\+\s*['\"](\w+)['\"]\]\s*=")
    litpat = re.compile(r"^\s*['\"](\w+)['\"]\s*:", re.M)
    fpat = re.compile(r"info\[f['\"]([\w{}]+)['\"]\]\s*=")
    rpat = re.compile(r"info(?:\.get\(|\[)['\"](\w+)['\"]")
    for p in list(SRC.rglob('*.py')) + [EXAMPLE]:
        s = p.read_text()
        writes.update(wpat.findall(s))
        dynamic_suffixes.update(dynpat.findall(s))
        for m in fpat.findall(s):
            dynamic_suffixes.add(m.split('}')[-1])   # f'cp_hold_{reason}'
            dynamic_suffixes.add(m.split('{')[0])
        if 'make_epoch_diagnostics' in s or 'epoch_context' in p.name:
            writes.update(litpat.findall(s))
        reads.update(rpat.findall(s))
    return writes, dynamic_suffixes, reads


def test_no_orphan_info_reads():
    writes, suffixes, reads = _collect()
    orphans = set()
    for k in reads - writes:
        if any(suf and (k.endswith(suf) or k.startswith(suf)) for suf in suffixes):
            continue
        orphans.add(k)
    assert not orphans, f"info keys read but never produced: {sorted(orphans)}"
