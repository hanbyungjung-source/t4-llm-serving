"""P6 dry run: the production kaggle_server module end to end inside the research session."""
import importlib.util
import json
import os
import pathlib
import secrets
import shutil
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path('/kaggle/temp/t4-engine')
sys.path.insert(0, str(ROOT))
import bench  # noqa: E402

spec = importlib.util.spec_from_file_location('ks_candidate', ROOT / 'prod' / 'kaggle_server.py')
ks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ks)
ks.ROOT = ROOT / 'prod' / 'root'
models = ks.ROOT / 'models'
models.mkdir(parents=True, exist_ok=True)
for name in (ks.MODEL_FILE, ks.PROJECTOR_FILE):
    if not (models / name).exists():
        (models / name).symlink_to(ROOT / 'models' / name)
started = time.monotonic()
ks.prepare()
print('PREPARE_SECONDS', round(time.monotonic()-started), flush=True)

key = secrets.token_hex(24)
os.environ['LOCAL_DESK_API_KEY'] = key
out = bench.RESULTS / 'p6-prod-dryrun'
out.mkdir(parents=True, exist_ok=False)
result = {'module_sha256': bench.hashlib.sha256((ROOT / 'prod' / 'kaggle_server.py').read_bytes()).hexdigest(),
          'command': ks.server_command(), 'samples': [], 'error': None}
sampler = bench.Sampler(out / 'gpu.csv')
process = None
try:
    started = time.monotonic()
    process = ks.start_server()
    result['load_seconds'] = time.monotonic()-started
    try:
        urllib.request.build_opener().open('http://127.0.0.1:8080/v1/models', timeout=10)
        result['unauthenticated'] = 'accepted'
    except urllib.error.HTTPError as error:
        result['unauthenticated'] = error.code
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'Bearer '+key)]
    urllib.request.install_opener(opener)
    for case in bench.load_cases():
        row = bench.request(case)
        result['samples'].append(row)
        print('CASE p6-prod-dryrun', row['name'], round(row['seconds'], 3), row['passed'], flush=True)
except Exception as error:
    result['error'] = '%s: %s' % (type(error).__name__, error)
finally:
    sampler.close()
    if process is not None and process.poll() is None:
        process.terminate()
        process.wait(timeout=60)
    result['gpu'] = bench.gpu_summary(out / 'gpu.csv')
    if result['samples']:
        result['summary'] = bench.summarize(result['samples'])
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    if (ks.ROOT / 'server.log').exists():
        shutil.copy(ks.ROOT / 'server.log', out / 'server.log')
    print('SUMMARY p6-prod-dryrun', json.dumps(result.get('summary')), result['error'],
          result.get('unauthenticated'), flush=True)
