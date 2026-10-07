"""Kaggle-side helpers for the T4 engine study: background jobs, logs, hashes."""
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path('/kaggle/temp/t4-engine')
LOGS = ROOT / 'logs'
JOBS = ROOT / 'jobs'


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        state = Path('/proc/%d/stat' % pid).read_text().split(')')[-1].split()[0]
    except OSError:
        return False
    return state != 'Z'


def start(name, command, cwd=None, env=None):
    LOGS.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    record = JOBS / (name+'.json')
    if record.exists() and alive(json.loads(record.read_text())['pid']):
        raise RuntimeError('job already running: '+name)
    exit_file = JOBS / (name+'.exit')
    exit_file.unlink(missing_ok=True)
    text = command if isinstance(command, str) else shlex.join(command)
    wrapped = 'set -o pipefail; ( %s ); echo $? > %s' % (text, shlex.quote(str(exit_file)))
    merged = dict(os.environ)
    merged.update(env or {})
    with (LOGS / (name+'.log')).open('ab') as log:
        process = subprocess.Popen(['bash', '-c', wrapped], cwd=cwd, env=merged, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    record.write_text(json.dumps({'pid': process.pid, 'command': text, 'started': time.time()}))
    return process.pid


def status(name, tail=2000):
    record = JOBS / (name+'.json')
    if not record.exists():
        return {'name': name, 'state': 'missing'}
    data = json.loads(record.read_text())
    exit_file = JOBS / (name+'.exit')
    code = int(exit_file.read_text().strip() or -1) if exit_file.exists() else None
    running = code is None and alive(data['pid'])
    log = LOGS / (name+'.log')
    text = ''
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-tail))
            text = stream.read().decode('utf-8', 'replace')
    return {'name': name, 'state': 'running' if running else ('exited' if code is not None else 'lost'),
            'exit': code, 'elapsed': round(time.time()-data['started'], 1), 'tail': text}


def stop(name):
    data = json.loads((JOBS / (name+'.json')).read_text())
    if alive(data['pid']):
        os.killpg(data['pid'], 15)
    return status(name, 0)


def sh(command, timeout=600):
    result = subprocess.run(command, shell=isinstance(command, str), capture_output=True,
                            text=True, timeout=timeout)
    return (result.stdout+result.stderr)[-20000:], result.returncode


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()
