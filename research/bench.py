"""Frozen-input llama-server benchmark for the T4 engine study (Kaggle side)."""
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path('/kaggle/temp/t4-engine')
MODELS = ROOT / 'models'
RESULTS = ROOT / 'results'


def input_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_cases():
    cases = json.loads((ROOT / 'cases.json').read_text())
    for case in cases:
        if input_hash(case['payload']) != case['input_sha256']:
            raise ValueError('Frozen case changed: '+case['name'])
    return cases


def server_command(binary, *, split='layer', tensor_split='36,30', draft=3, context=196608,
                   draft_device='CUDA1', extra=(), draft_model=None):
    command = [str(binary), '--model', str(MODELS / 'Qwen3.8-27B-UD-Q4_K_XL.gguf'),
               '--mmproj', str(MODELS / 'mmproj-F16.gguf'), '--alias', 'qwen3.8-27b-ud-q4-k-xl',
               '--host', '127.0.0.1', '--port', '8080', '--device', 'CUDA0,CUDA1',
               '--split-mode', split, '--n-gpu-layers', 'all', '--ctx-size', str(context),
               '--parallel', '1', '--batch-size', '512', '--ubatch-size', '128',
               '--threads', '4', '--threads-batch', '4', '--flash-attn', 'on',
               '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0']
    if tensor_split:
        command += ['--tensor-split', tensor_split]
    if draft:
        command += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', str(draft)]
        if draft_model != 'self':
            command += ['--spec-draft-model', str(draft_model or MODELS / 'MTP' / 'mtp-Qwen3.8-27B-Q4_0.gguf'),
                        '--spec-draft-device', draft_device, '--spec-draft-ngl', 'all']
    command += ['--jinja', '--chat-template-kwargs', '{"reasoning_effort":"xhigh"}',
                '--temp', '1.0', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.0',
                '--n-predict', '-1', '--metrics', '--verbosity', '4', '--fit', 'off']
    return command + list(extra)


def wait_ready(process, timeout=600):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('server exited with %s' % process.returncode)
        try:
            with urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3) as response:
                if json.load(response).get('status') == 'ok':
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError('server not ready')


def request(case, timeout=180):
    text, finish, usage, timings = '', None, {}, {}
    started = time.monotonic()
    first = None
    call = urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions',
                                  data=json.dumps(case['payload']).encode(),
                                  headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(call, timeout=timeout) as response:
        for raw in response:
            if not raw.startswith(b'data:'):
                continue
            data = raw[5:].strip()
            if data == b'[DONE]':
                break
            event = json.loads(data)
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                if first is None and (delta.get('content') or delta.get('reasoning_content')):
                    first = time.monotonic()-started
                text += delta.get('content') or ''
                finish = choice.get('finish_reason') or finish
            usage.update(event.get('usage') or {})
            timings.update(event.get('timings') or {})
    seconds = time.monotonic()-started
    try:
        action = json.loads(text)
        passed = (finish == 'stop' and isinstance(action, dict) and set(action) == {'tool', 'arguments'}
                  and action['tool'] == 'finish' and isinstance(action['arguments'], dict)
                  and not set(action['arguments'])-{'text', 'commentary'}
                  and action['arguments'].get('text') == case['expected'])
    except ValueError:
        passed = False
    return {'name': case['name'], 'input_sha256': case['input_sha256'], 'passed': passed,
            'seconds': seconds, 'first_token_seconds': first, 'usage': usage, 'timings': timings,
            'answer_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'answer_text': None if passed else text[:400]}


class Sampler:
    def __init__(self, path):
        self.stream = open(path, 'w')
        self.process = subprocess.Popen(
            ['nvidia-smi', '--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,'
             'clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active',
             '--format=csv,noheader,nounits', '-lms', '250'], stdout=self.stream,
            stderr=subprocess.DEVNULL)

    def close(self):
        self.process.terminate()
        self.process.wait(timeout=10)
        self.stream.close()


def gpu_summary(path):
    rows = [line.split(', ') for line in Path(path).read_text().splitlines() if line.count(',') >= 9]
    summary = {}
    for index in ('0', '1'):
        own = [row for row in rows if row[1] == index]
        if not own:
            continue
        used = [float(row[2]) for row in own]
        summary[index] = {'samples': len(own), 'max_used_mib': max(used),
                          'free_min_mib': min(float(row[3])-float(row[2]) for row in own),
                          'mean_util': sum(float(row[4]) for row in own)/len(own),
                          'mean_sm_mhz': sum(float(row[5]) for row in own)/len(own),
                          'min_sm_mhz': min(float(row[5]) for row in own),
                          'max_power_w': max(float(row[7]) for row in own),
                          'max_temp_c': max(float(row[8]) for row in own)}
    return summary


def summarize(samples):
    scored = [row for row in samples if row['name'] != 'warmup']
    ms = sum(row['timings'].get('predicted_ms', 0) for row in scored)
    tokens = sum(max(0, row['timings'].get('predicted_n', 0)-1) for row in scored)
    drafted = sum(row['timings'].get('draft_n', 0) for row in scored)
    accepted = sum(row['timings'].get('draft_n_accepted', 0) for row in scored)
    return {'all_passed': all(row['passed'] for row in samples),
            'scored_seconds': sum(row['seconds'] for row in scored),
            'decode_tps': 1000*tokens/ms if ms else None,
            'completion_tokens': [row['usage'].get('completion_tokens') for row in scored],
            'prompt_tokens': [row['usage'].get('prompt_tokens') for row in scored],
            'draft_n': drafted, 'draft_accepted': accepted,
            'case_seconds': {row['name']: round(row['seconds'], 3) for row in scored},
            'prompt_ms': {row['name']: row['timings'].get('prompt_ms') for row in scored}}


def run(name, command, *, env=None, cases=None, profile=None):
    """Start one server, run the frozen cases once, stop it. profile=(case_name, nsys_output)."""
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=False)
    cases = cases or load_cases()
    merged = dict(os.environ)
    merged.update(env or {})
    launch = list(command)
    session = None
    if profile:
        session = 't4'+str(int(time.time()))
        launch = ['nsys', 'launch', '--session-new='+session, '--trace=cuda,osrt,nvtx',
                  '--cuda-graph-trace=node', '--wait=primary'] + launch
    result = {'name': name, 'command': command, 'env': env or {}, 'samples': [], 'error': None}
    log = (out / 'server.log').open('w')
    process = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT, env=merged,
                               start_new_session=True)
    sampler = Sampler(out / 'gpu.csv')
    try:
        started = time.monotonic()
        wait_ready(process)
        result['load_seconds'] = time.monotonic()-started
        for case in cases:
            capture = profile and case['name'] == profile[0]
            if capture:
                subprocess.run(['nsys', 'start', '--session='+session, '--output='+str(profile[1]),
                                '--force-overwrite=true'], check=True, timeout=60)
            try:
                row = request(case)
            finally:
                if capture:
                    subprocess.run(['nsys', 'stop', '--session='+session], check=True, timeout=600)
            result['samples'].append(row)
            print('CASE', name, row['name'], round(row['seconds'], 3), row['passed'], flush=True)
    except Exception as error:
        result['error'] = '%s: %s' % (type(error).__name__, error)
    finally:
        sampler.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        log.close()
        result['gpu'] = gpu_summary(out / 'gpu.csv')
        if result['samples']:
            result['summary'] = summarize(result['samples'])
        with (out / 'results.json').open('x') as stream:
            json.dump(result, stream, indent=2)
    return result
