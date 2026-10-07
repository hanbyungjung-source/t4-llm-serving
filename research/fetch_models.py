"""Download the study models with pinned revision and SHA-256 checks (Kaggle side)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

MODEL_REPO = 'unsloth/Qwen3.8-27B-GGUF'
MODEL_REVISION = '4ca720788d1e01f1bff70c033e0d0028fd02e502'
FILES = ('Qwen3.8-27B-UD-Q4_K_XL.gguf', 'MTP/mtp-Qwen3.8-27B-Q4_0.gguf', 'mmproj-F16.gguf')
MODELS = Path('/kaggle/temp/t4-engine/models')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub==0.34.4'], check=True, timeout=300)
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(MODEL_REPO, revision=MODEL_REVISION, files_metadata=True)
    siblings = {item.rfilename: item for item in info.siblings}
    manifest = {'repo': MODEL_REPO, 'revision': info.sha, 'files': {}}
    for name in FILES:
        expected = siblings[name]
        path = Path(hf_hub_download(MODEL_REPO, name, revision=MODEL_REVISION, local_dir=MODELS))
        digest = sha256(path)
        if path.stat().st_size != expected.size or digest != expected.lfs.sha256:
            raise SystemExit('integrity check failed: '+name)
        manifest['files'][name] = {'bytes': expected.size, 'sha256': digest}
        print('VERIFIED', name, digest, flush=True)
    (MODELS / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
