"""Build a long-context needle case (~25K tokens) from the frozen warmup case (T4 study)."""
import copy
import json
import sys

sys.path.insert(0, '/kaggle/temp/t4-engine')
import bench

WORDS = ['apple', 'river', 'stone', 'cloud', 'maple', 'tiger', 'ember', 'lotus', 'amber', 'cedar', 'delta', 'frost',
         'grape', 'harbor', 'ivory', 'jasper', 'kiwi', 'lemon', 'mango', 'nectar', 'olive', 'pearl', 'quartz', 'raven']
LINES = 4000
NEEDLE = 2718


def main():
    template = next(case for case in json.loads((bench.ROOT / 'cases.json').read_text()) if case['name'] == 'warmup')
    words = {index: WORDS[(index*7+3) % len(WORDS)] for index in range(1, LINES+1)}
    body = '\n'.join('L%05d %s' % (index, word) for index, word in words.items())
    payload = copy.deepcopy(template['payload'])
    payload['messages'][-1]['content'] = ('Below is a numbered list.\n'+body+'\n\nReturn only the word on line L%05d, '
                                          'with no explanation.' % NEEDLE)
    case = {'name': 'long', 'payload': payload, 'expected': words[NEEDLE], 'input_sha256': bench.input_hash(payload)}
    (bench.ROOT / 'cases_long.json').write_text(json.dumps([case], ensure_ascii=False))
    print('long case chars', len(payload['messages'][-1]['content']), 'expected', words[NEEDLE])


if __name__ == '__main__':
    main()
