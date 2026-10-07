"""Compare two test-backend-ops perf logs (T4 study). Usage: python parse_ops.py <a.txt> <b.txt> [out.json]"""
import json
import re
import sys

ANSI = re.compile(r'\x1b\[[0-9;]*m')
LINE = re.compile(r'MUL_MAT\(type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+),[^)]*\):\s*(\d+) runs -\s*([\d.]+) us/run')
BITS = {'q3_K': 3.4375, 'q4_K': 4.5, 'q5_K': 5.5, 'q6_K': 6.5625, 'iq4_xs': 4.25, 'iq4_nl': 4.5,
        'q4_0': 4.5, 'q8_0': 8.5}


def parse(path):
    rows = {}
    for raw in open(path, errors='replace'):
        match = LINE.search(ANSI.sub('', raw))
        if match:
            kind, m, n, k, _, us = match.groups()
            weight_bytes = int(m)*int(k)*BITS.get(kind, 0)/8
            rows['%s m=%s k=%s n=%s' % (kind, m, k, n)] = {'us': float(us), 'gbs': weight_bytes/float(us)/1e3}
    return rows


def main(first, second, out=None):
    a, b = parse(first), parse(second)
    table = []
    for key in sorted(set(a) & set(b)):
        table.append({'case': key, 'a_us': a[key]['us'], 'b_us': b[key]['us'], 'a_gbs': round(a[key]['gbs'], 1),
                      'b_gbs': round(b[key]['gbs'], 1), 'speedup_b': round(a[key]['us']/b[key]['us'], 3)})
    for row in table:
        print('%-30s a %8.1fus %6.1fGB/s | b %8.1fus %6.1fGB/s | x%.2f' % (
            row['case'], row['a_us'], row['a_gbs'], row['b_us'], row['b_gbs'], row['speedup_b']))
    print('rows', len(table), 'only_a', len(set(a)-set(b)), 'only_b', len(set(b)-set(a)))
    if out:
        with open(out, 'w') as stream:
            json.dump(table, stream, indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:4])
