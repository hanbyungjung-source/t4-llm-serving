"""Build the extended exact-answer functional set from the frozen warmup case (T4 study)."""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, '/kaggle/temp/t4-engine')
import bench

TASKS = [
    ('Return the number of letters in PLANET as text, with no explanation.', '6'),
    ('Return the number of letters in KAGGLE as text, with no explanation.', '6'),
    ('Compute 12 times 7. Return only the number as text.', '84'),
    ('Compute 144 divided by 12. Return only the number as text.', '12'),
    ('Compute (25 + 17) * 2. Return only the number as text.', '84'),
    ('Compute 3 to the power of 4. Return only the number as text.', '81'),
    ('Compute 1000 - 387. Return only the number as text.', '613'),
    ('Convert "blue sky" to uppercase. Return only the converted text.', 'BLUE SKY'),
    ('Reverse the letters of "stone". Return only the reversed word.', 'enots'),
    ('Return the initials of "Graphics Processing Unit" in uppercase, with no explanation.', 'GPU'),
    ('Sort the letters c, a, b alphabetically and return them joined by commas without spaces.', 'a,b,c'),
    ('Count the vowels (a, e, i, o, u) in "education". Return only the number as text.', '5'),
    ('What is the smallest prime number greater than 13? Return only the number as text.', '17'),
    ('"사과"는 한글 몇 글자입니까? 숫자만 답하세요.', '2'),
    ('3 더하기 4는 얼마입니까? 숫자만 답하세요.', '7'),
    ('Which day comes right after Monday? Return only the English day name.', 'Tuesday'),
    ('What is 15 percent of 200? Return only the number as text.', '30'),
    ('Convert the binary number 1010 to decimal. Return only the number as text.', '10'),
    ('Which is smaller, 0.9 or 0.11? Return only the smaller number as text.', '0.11'),
    ('How many days are in a leap-year February? Return only the number as text.', '29'),
]


def main():
    template = next(case for case in json.loads((bench.ROOT / 'cases.json').read_text()) if case['name'] == 'warmup')
    cases = []
    for index, (prompt, expected) in enumerate(TASKS):
        payload = copy.deepcopy(template['payload'])
        payload['messages'][-1]['content'] = prompt
        cases.append({'name': 'ext%02d' % index, 'payload': payload, 'expected': expected,
                      'input_sha256': bench.input_hash(payload)})
    Path(bench.ROOT / 'cases_ext.json').write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    print('cases', len(cases))


if __name__ == '__main__':
    main()
