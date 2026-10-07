"""Compare two test-backend-ops perf logs (T4 study). Usage: python parse_ops.py <a.txt> <b.txt> [out.json]"""
import json
import re
import sys

ANSI = re.compile(r'\x1b\[[0-9;]*m')
LINE = re.compile(r'MUL_MAT\(type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+),[^)]*\):\s*(\d+) runs -\s*([\d.]+) us/run -'
                  r'\s*(\d+) kB/run -\s*([\d.]+) GB/s')


def parse(path):
    rows = {}
    for raw in open(path, errors='replace'):
        match = LINE.search(ANSI.sub('', raw))
        if match:
            kind, m, n, k, _, us, _, gbs = match.groups()
            rows['%s m=%s k=%s n=%s' % (kind, m, k, n)] = {'us': float(us), 'gbs': float(gbs)}
    return rows


def main(first, second, out=None):
    a, b = parse(first), parse(second)
    table = []
    for key in sorted(set(a) & set(b)):
        table.append({'case': key, 'a_us': a[key]['us'], 'b_us': b[key]['us'], 'a_gbs': a[key]['gbs'],
                      'b_gbs': b[key]['gbs'], 'speedup_b': round(a[key]['us']/b[key]['us'], 3)})
    for row in table:
        print('%-34s a %8.1fus %6.1fGB/s | b %8.1fus %6.1fGB/s | x%.2f' % (
            row['case'], row['a_us'], row['a_gbs'], row['b_us'], row['b_gbs'], row['speedup_b']))
    print('rows', len(table), 'only_a', len(set(a)-set(b)), 'only_b', len(set(b)-set(a)))
    if out:
        with open(out, 'w') as stream:
            json.dump(table, stream, indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:4])
