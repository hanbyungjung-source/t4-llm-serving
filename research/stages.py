"""Stage runner for the T4 engine study (Kaggle side). Usage: python stages.py <stage> [<stage> ...]"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, '/kaggle/temp/t4-engine')
import analyze_nsys
import bench

ROOT = Path('/kaggle/temp/t4-engine')
RESULTS = ROOT / 'results'
MODEL = ROOT / 'models' / 'Qwen3.8-27B-UD-Q4_K_XL.gguf'
os.environ['PATH'] = '/opt/nvidia/nsight-systems/2026.3.2/bin:'+os.environ['PATH']


def binary(tag, name):
    return ROOT / 'src' / tag / 'build' / 'bin' / name


def llama_bench(tag, name, args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    command = [str(binary(tag, 'llama-bench')), '-m', str(MODEL), '-ngl', '99', '-fa', '1',
               '-ctk', 'q8_0', '-ctv', 'q8_0', '-b', '512', '-ub', '128', '-o', 'json'] + args
    process = subprocess.run(command, capture_output=True, text=True, timeout=7200)
    (RESULTS / (name+'.stderr.txt')).write_text(process.stderr[-50000:])
    with (RESULTS / (name+'.json')).open('x') as stream:
        stream.write(process.stdout)
    rows = json.loads(process.stdout) if process.returncode == 0 and process.stdout.strip() else []
    for row in rows:
        print('BENCH', name, row.get('split_mode'), row.get('n_prompt'), row.get('n_gen'),
              row.get('n_depth'), round(row.get('avg_ts', 0), 3), flush=True)
    return process.returncode


MODEL_SHAPES = 'm=(17408|10240|6144|12288|62080),n=[14],k=5120,|m=5120,n=[14],k=(17408|6144),'


def ops(tag, name, regex=None, env=None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    regex = regex or 'type_a=(q3_K|q4_K|q5_K|q6_K|q8_0|iq4_xs|iq4_nl|q4_0),type_b=f32,m=4096,n=(1|2|3|4|5|8),k=14336,'
    command = [str(binary(tag, 'test-backend-ops')), 'perf', '-o', 'MUL_MAT', '-b', 'CUDA0', '-p', regex]
    merged = dict(os.environ)
    merged.update(env or {})
    process = subprocess.run(command, capture_output=True, text=True, timeout=3600, env=merged)
    (RESULTS / (name+'.txt')).write_text(process.stdout[-200000:]+'\n--- stderr ---\n'+process.stderr[-20000:])
    for line in process.stdout.splitlines():
        if 'MUL_MAT(' in line:
            print('OPS', line.strip(), flush=True)
    return process.returncode


def op_eval(tag, name, env=None, op='MUL_MAT'):
    RESULTS.mkdir(parents=True, exist_ok=True)
    merged = dict(os.environ)
    merged.update(env or {})
    command = [str(binary(tag, 'test-backend-ops')), 'test', '-o', op, '-b', 'CUDA0']
    process = subprocess.run(command, capture_output=True, text=True, timeout=3600, env=merged)
    (RESULTS / (name+'.txt')).write_text(process.stdout[-400000:]+'\n--- stderr ---\n'+process.stderr[-20000:])
    failed = [line.strip() for line in process.stdout.splitlines() if 'FAIL' in line]
    passed = sum('OK' in line for line in process.stdout.splitlines())
    print('EVAL', name, 'rc', process.returncode, 'ok', passed, 'fail', len(failed), failed[:5], flush=True)
    return process.returncode


CORPUS = ROOT / 'data' / 'wikitext-2-raw' / 'wiki.test.raw'
KLD_BASE = RESULTS / 'kld-base-v050.bin'
KLD_KEYS = ('Mean PPL', 'Mean    KLD', 'Maximum KLD', '99.9%   KLD', '99.0%   KLD', 'Median  KLD', 'Same top p',
            'RMS Δp', 'Mean    Δp')


def corpus():
    if not CORPUS.exists():
        import io
        import urllib.request
        import zipfile
        url = 'https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip'
        with urllib.request.urlopen(url, timeout=300) as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extractall(ROOT / 'data')
    return CORPUS


def kld(tag, name, env=None, batch=4, ubatch=4, save=False, chunks=8, split='layer'):
    RESULTS.mkdir(parents=True, exist_ok=True)
    placement = ['-sm', 'layer', '-ts', '36,30'] if split == 'layer' else ['-sm', split]
    command = [str(binary(tag, 'llama-perplexity')), '-m', str(MODEL), '-c', '512', '--chunks', str(chunks),
               '-b', str(batch), '-ub', str(ubatch), '-ngl', '99'] + placement + ['-fa', 'on',
               '-ctk', 'q8_0', '-ctv', 'q8_0', '--kl-divergence-base', str(KLD_BASE)]
    command += ['-f', str(corpus())] if save else ['--kl-divergence']
    merged = dict(os.environ)
    merged.update(env or {})
    process = subprocess.run(command, capture_output=True, text=True, timeout=7200, env=merged)
    (RESULTS / (name+'.txt')).write_text(process.stdout[-100000:]+'\n--- stderr ---\n'+process.stderr[-100000:])
    for line in (process.stdout+process.stderr).splitlines():
        if any(key in line for key in KLD_KEYS):
            print('KLD', name, line.strip(), flush=True)
    return process.returncode


def cool_down(limit_c=55, timeout=420):
    started = time.monotonic()
    while True:
        output = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True).stdout.split()
        temps = [int(value) for value in output]
        waited = round(time.monotonic()-started, 1)
        if max(temps) <= limit_c or waited > timeout:
            return temps, waited
        time.sleep(10)


def cases(tag, name, profile_case=None, env=None, ext=False, **server):
    temps, waited = cool_down()
    print('COOL', name, temps, waited, flush=True)
    command = bench.server_command(binary(tag, 'llama-server'), **server)
    profile = (profile_case, str(RESULTS / name / profile_case)) if profile_case else None
    chosen = json.loads((ROOT / ('cases_%s.json' % ext)).read_text()) if ext else None
    result = bench.run(name, command, env=env, cases=chosen, profile=profile)
    print('SUMMARY', name, json.dumps(result.get('summary')), result.get('error'), flush=True)
    if profile and (RESULTS / name / (profile_case+'.nsys-rep')).exists():
        analysis = analyze_nsys.analyze(RESULTS / name / (profile_case+'.nsys-rep'))
        (RESULTS / name / 'nsys-analysis.json').write_text(json.dumps(analysis, indent=2))
        print('NSYS', name, json.dumps(analysis)[:3000], flush=True)
    return 0 if result.get('summary', {}).get('all_passed') and not result.get('error') else 1


STAGES = {
    'b11000-bench': lambda: llama_bench('b11000', 'p1-llamabench-b11000-layer2',
                                        ['-sm', 'layer', '-ts', '36/30', '-p', '512', '-n', '64', '-r', '2']),
    'b11000-profile': lambda: cases('b11000', 'p1-profile-b11000', profile_case='copy'),
    'b11000-ops': lambda: ops('b11000', 'p1-ops-b11000'),
    'b11000-base': lambda: cases('b11000', 'p1-b11000-base'),
    'b11000-shapes-mmvq': lambda: ops('b11000', 'p2-shapes-b11000-default', regex=MODEL_SHAPES),
    'b11000-shapes-mmq': lambda: ops('b11000', 'p2-shapes-b11000-mmq', regex=MODEL_SHAPES,
                                     env={'GGML_CUDA_MMVQ_MAX_BATCH': '0'}),
    'b11000-mmq1': lambda: cases('b11000', 'p2-b11000-mmq1', env={'GGML_CUDA_MMVQ_MAX_BATCH': '1'}),
    'b11000-eval-mmq': lambda: op_eval('b11000', 'p2-eval-b11000-mmq', env={'GGML_CUDA_MMVQ_MAX_BATCH': '0'}),
    'b11000-selfmtp': lambda: cases('b11000', 'p2-b11000-selfmtp', draft_model='self'),
    'b11000-headq40': lambda: cases('b11000', 'p2-b11000-headq40',
                                    draft_model=ROOT / 'models' / 'MTP' / 'mtp-head-Q4_0.gguf'),
    'b11000-headq80': lambda: cases('b11000', 'p2-b11000-headq80',
                                    draft_model=ROOT / 'models' / 'MTP' / 'mtp-head-Q8_0.gguf'),
    'v050-tensor-selfmtp': lambda: cases('v050', 'p2-v050-tensor-selfmtp', split='tensor', tensor_split=None,
                                         draft_model='self'),
    'v050-tensor-headq40': lambda: cases('v050', 'p2-v050-tensor-headq40', split='tensor', tensor_split=None,
                                         draft_model=ROOT / 'models' / 'MTP' / 'mtp-head-Q4_0.gguf'),
    'kld-base': lambda: kld('v050', 'p2-kld-base', save=True),
    'kld-mmq1': lambda: kld('v050', 'p2-kld-mmq1', env={'GGML_CUDA_MMVQ_MAX_BATCH': '1'}),
    'kld-b512': lambda: kld('v050', 'p2-kld-b512', batch=512, ubatch=128),
    'kld-tensor': lambda: kld('v050', 'p2-kld-tensor', split='tensor'),
    'kld-tensor-mmq1': lambda: kld('v050', 'p2-kld-tensor-mmq1', split='tensor', env={'GGML_CUDA_MMVQ_MAX_BATCH': '1'}),
    'v050-layer-mmq1': lambda: cases('v050', 'p2-v050-layer-mmq1', env={'GGML_CUDA_MMVQ_MAX_BATCH': '1'}),
    'v050-tensor-mmq1': lambda: cases('v050', 'p2-v050-tensor-mmq1', split='tensor', tensor_split=None,
                                      env={'GGML_CUDA_MMVQ_MAX_BATCH': '1'}),
    'v050-tensor-selfmtp-mmq1': lambda: cases('v050', 'p2-v050-tensor-selfmtp-mmq1', split='tensor',
                                              tensor_split=None, draft_model='self',
                                              env={'GGML_CUDA_MMVQ_MAX_BATCH': '1'}),
    'v050-layer-ref2': lambda: cases('v050', 'p2-v050-layer-ref2'),
    'ext-b11000-base': lambda: cases('b11000', 'p5-ext-b11000-base', ext='ext'),
    'ext-v050-layer': lambda: cases('v050', 'p5-ext-v050-layer', ext='ext'),
    'ext-v050-tensor-selfmtp': lambda: cases('v050', 'p5-ext-v050-tensor-selfmtp', ext='ext', split='tensor',
                                             tensor_split=None, draft_model='self'),
    'ext-v050-tensor-selfmtp-mmq1': lambda: cases('v050', 'p5-ext-v050-tensor-selfmtp-mmq1', ext='ext',
                                                  split='tensor', tensor_split=None, draft_model='self',
                                                  env={'GGML_CUDA_MMVQ_MAX_BATCH': '1'}),
    'long-b11000-base': lambda: cases('b11000', 'p5-long-b11000-base', ext='long'),
    'long-v050-final': lambda: cases('v050', 'p5-long-v050-final', ext='long', split='tensor', tensor_split=None,
                                     draft_model='self', env={'GGML_CUDA_MMVQ_MAX_BATCH': '1'}),
    'b11000-base2': lambda: cases('b11000', 'p2-b11000-base2'),
    'v050-bench-layer': lambda: llama_bench('v050', 'p2-llamabench-v050-layer',
                                            ['-sm', 'layer', '-ts', '36/30', '-p', '512', '-n', '64', '-r', '2']),
    'v050-bench-tensor': lambda: llama_bench('v050', 'p2-llamabench-v050-tensor',
                                             ['-sm', 'tensor', '-p', '512', '-n', '64', '-r', '2']),
    'v050-layer': lambda: cases('v050', 'p2-v050-layer'),
    'v050-tensor': lambda: cases('v050', 'p2-v050-tensor', split='tensor', tensor_split=None),
    'v050-tensor-profile': lambda: cases('v050', 'p2-v050-tensor-profile', profile_case='copy',
                                         split='tensor', tensor_split=None),
}


if __name__ == '__main__':
    failures = 0
    for stage in sys.argv[1:]:
        print('STAGE', stage, flush=True)
        code = STAGES[stage]()
        print('STAGE_DONE', stage, code, flush=True)
        failures += code != 0
    sys.exit(1 if failures else 0)
