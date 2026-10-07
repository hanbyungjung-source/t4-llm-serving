"""Decompose an nsys capture: per-GPU busy/idle, overlap, kernel categories, copies, CPU sync calls."""
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

CATEGORIES = [
    ('mmvq', r'mul_mat_vec_q'), ('mmq', r'mul_mat_q|mmq'), ('mmf', r'mul_mat_vec_f|mul_mat_f|mmf'),
    ('cublas', r'gemm|cutlass|cublas|xmma'), ('fattn', r'flash_attn|fattn'),
    ('gdn', r'gated_delta|delta_net|ssm_conv|ssm_scan|rwkv|gla'), ('norm', r'norm'),
    ('rope', r'rope'), ('sample', r'soft_max|argsort|top_k|argmax|cumsum'),
    ('allreduce', r'allreduce|_ar_'), ('quantize', r'quantize'),
    ('copy', r'cpy|copy|concat|get_rows|set_rows|dup|convert'),
    ('elementwise', r'add|mul|scale|silu|sigmoid|glu|sub|div|exp|softplus|clamp|unary|binary|fill'),
]


def union(intervals):
    total, end = 0, None
    start = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            if end is not None:
                total += end-start
            start, end = a, b
        else:
            end = max(end, b)
    if end is not None:
        total += end-start
    return total, merge(intervals)


def merge(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def intersect(x, y):
    i = j = total = 0
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if a < b:
            total += b-a
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return total


def category(name):
    lowered = name.lower()
    for label, pattern in CATEGORIES:
        if re.search(pattern, lowered):
            return label
    return 'other'


def analyze(report):
    report = Path(report)
    database = report.with_suffix('.sqlite')
    if not database.exists():
        subprocess.run(['nsys', 'export', '--type=sqlite', '--output='+str(database), str(report)],
                       check=True, timeout=1200, capture_output=True)
    db = sqlite3.connect(database)
    names = dict(db.execute('select id, value from StringIds'))
    tables = {row[0] for row in db.execute("select name from sqlite_master where type='table'")}
    activity = {}
    kernels = {}
    counts = {}
    for start, end, device, short in db.execute(
            'select start, end, deviceId, shortName from CUPTI_ACTIVITY_KIND_KERNEL'):
        activity.setdefault(device, []).append((start, end))
        label = category(names.get(short, ''))
        kernels.setdefault(device, {}).setdefault(label, 0)
        kernels[device][label] += end-start
        counts[device] = counts.get(device, 0)+1
    copies = {}
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        kinds = {1: 'HtoD', 2: 'DtoH', 8: 'DtoD', 10: 'PtoP'}
        for start, end, device, kind, size in db.execute(
                'select start, end, deviceId, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY'):
            activity.setdefault(device, []).append((start, end))
            key = kinds.get(kind, str(kind))
            entry = copies.setdefault(str(device)+':'+key, {'count': 0, 'ns': 0, 'bytes': 0})
            entry['count'] += 1
            entry['ns'] += end-start
            entry['bytes'] += size
    everything = [interval for rows in activity.values() for interval in rows]
    window = (min(a for a, _ in everything), max(b for _, b in everything))
    span = window[1]-window[0]
    busy = {}
    merged = {}
    for device, rows in activity.items():
        busy[device], merged[device] = union(rows)
    any_busy, _ = union(everything)
    devices = sorted(merged)
    both = intersect(merged[devices[0]], merged[devices[1]]) if len(devices) == 2 else 0
    api = {}
    if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables:
        for start, end, name in db.execute('select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
            label = re.sub(r'_v\d+$', '', names.get(name, str(name)))
            entry = api.setdefault(label, {'count': 0, 'ns': 0})
            entry['count'] += 1
            entry['ns'] += end-start
    top_api = dict(sorted(api.items(), key=lambda item: -item[1]['ns'])[:12])
    ms = lambda value: round(value/1e6, 3)
    return {
        'report': report.name, 'window_ms': ms(span),
        'busy_ms': {str(d): ms(v) for d, v in busy.items()},
        'busy_fraction': {str(d): round(v/span, 4) for d, v in busy.items()},
        'both_busy_ms': ms(both), 'any_busy_ms': ms(any_busy), 'all_idle_ms': ms(span-any_busy),
        'kernel_count': {str(d): c for d, c in counts.items()},
        'kernel_ms': {str(d): {k: ms(v) for k, v in sorted(cats.items(), key=lambda item: -item[1])}
                      for d, cats in kernels.items()},
        'copies': {k: {'count': v['count'], 'ms': ms(v['ns']), 'mib': round(v['bytes']/2**20, 3)}
                   for k, v in copies.items()},
        'cuda_api_ms': {k: {'count': v['count'], 'ms': ms(v['ns'])} for k, v in top_api.items()},
    }


if __name__ == '__main__':
    print(json.dumps(analyze(sys.argv[1]), indent=2))
