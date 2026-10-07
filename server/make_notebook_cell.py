"""Build the Kaggle notebook cell that restores kaggle_server.py with a SHA-256 integrity check.

Usage: python make_notebook_cell.py [output.py]  (default: notebook_cell.py next to this file)
"""
import hashlib
import json
import sys
from pathlib import Path

source = Path(__file__).with_name('kaggle_server.py').read_bytes().decode('utf-8').replace('\r\n', '\n')
digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
cell = '\n'.join([
    'from pathlib import Path',
    'import hashlib',
    'import importlib.util',
    'import sys',
    '',
    '_SERVER_SOURCE = ' + json.dumps(source),
    '_SERVER_SHA256 = ' + json.dumps(digest),
    '',
    'def restore_server_module(directory="/kaggle/working"):',
    '    module_dir = Path(directory)',
    '    module_dir.mkdir(parents=True, exist_ok=True)',
    '    module_path = module_dir / "kaggle_server.py"',
    '    source_bytes = _SERVER_SOURCE.encode("utf-8")',
    '    if hashlib.sha256(source_bytes).hexdigest() != _SERVER_SHA256:',
    '        raise RuntimeError("Embedded server source integrity check failed")',
    '    module_path.write_bytes(source_bytes)',
    '    if hashlib.sha256(module_path.read_bytes()).hexdigest() != _SERVER_SHA256:',
    '        raise RuntimeError("Restored server source integrity check failed")',
    '    return module_path',
    '',
    'server_module_path = restore_server_module()',
    'if str(server_module_path.parent) not in sys.path:',
    '    sys.path.insert(0, str(server_module_path.parent))',
    'server_spec = importlib.util.spec_from_file_location("kaggle_server", server_module_path)',
    'ks = importlib.util.module_from_spec(server_spec)',
    'sys.modules[server_spec.name] = ks',
    'server_spec.loader.exec_module(ks)',
    'print("Server module restored: " + str(server_module_path))',
    'print("SHA256: " + _SERVER_SHA256)',
    '',
])
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('notebook_cell.py')
out.write_bytes(cell.encode('utf-8'))
print('embedded source sha256', digest, 'chars', len(source))
print('cell file', out, 'sha256', hashlib.sha256(cell.encode('utf-8')).hexdigest())
