import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
import kaggle_server  # noqa: E402


class KaggleServerTests(unittest.TestCase):
    def test_ngrok_failure_keeps_existing_model_and_does_not_publish_wrong_domain(self):
        import io
        import json
        import urllib.error
        for exit_code in (1, None):
            server, tunnel = Mock(), Mock(pid=4321)
            server.poll.return_value = None
            tunnel.poll.return_value = exit_code
            responses = [urllib.error.HTTPError('http://localhost',401,'',{},None),
                         urllib.error.HTTPError('http://localhost',403,'',{},None),
                         io.BytesIO(json.dumps({'data':[{'id':kaggle_server.ALIAS}]}).encode()),
                         io.BytesIO(json.dumps({'tunnels':[{'public_url':'https://wrong.ngrok-free.dev',
                                                           'config':{'addr':'http://127.0.0.1:8080'}}]}).encode())]
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as folder, \
                    patch.object(kaggle_server, 'ROOT', Path(folder)), \
                    patch.object(kaggle_server, 'prepare_ngrok', return_value=Path(folder)/'ngrok'), \
                    patch.object(kaggle_server, 'ngrok_domain', return_value='desk.ngrok-free.dev'), \
                    patch.object(kaggle_server, 'read_secret', return_value='test-secret'), \
                    patch.object(kaggle_server.socket, 'socket'), \
                    patch.object(kaggle_server.urllib.request, 'urlopen', side_effect=responses), \
                    patch.object(kaggle_server.subprocess, 'Popen', return_value=tunnel), \
                    patch('time.monotonic', side_effect=[0,0,31]), patch('time.sleep'):
                with self.assertRaises((RuntimeError, TimeoutError)) as caught:
                    kaggle_server.start_tunnel(server, allow_short_key=True)
                self.assertNotIn('test-secret', str(caught.exception))
                self.assertFalse((Path(folder)/'fixed-tunnel.json').exists())
                server.terminate.assert_not_called()
                self.assertEqual(tunnel.terminate.call_count, int(exit_code is None))

    def test_ngrok_tunnel_is_fixed_and_keeps_tokens_out_of_files_and_arguments(self):
        import io
        import json
        import urllib.error
        server, tunnel = Mock(), Mock(pid=4321)
        server.poll.return_value = tunnel.poll.return_value = None
        values = {'LOCAL_DESK_API_KEY':'test-api-key','NGROK_DOMAIN':'desk.ngrok-free.dev',
                  'NGROK_AUTHTOKEN':'test-ngrok-token-value-123456789012345'}
        responses = [urllib.error.HTTPError('http://localhost',401,'',{},None),
                     urllib.error.HTTPError('http://localhost',403,'',{},None),
                     io.BytesIO(json.dumps({'data':[{'id':kaggle_server.ALIAS}]}).encode()),
                     io.BytesIO(json.dumps({'tunnels':[{'public_url':'https://desk.ngrok-free.dev',
                                                       'config':{'addr':'http://127.0.0.1:8080'}}]}).encode())]
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(kaggle_server, 'ROOT', Path(folder)), \
                patch.object(kaggle_server, 'prepare_ngrok', return_value=Path(folder)/'ngrok'), \
                patch.object(kaggle_server, 'read_secret', side_effect=lambda label,**kwargs:values[label]), \
                patch.object(kaggle_server.socket, 'socket'), \
                patch.object(kaggle_server.urllib.request, 'urlopen', side_effect=responses), \
                patch.object(kaggle_server.subprocess, 'Popen', return_value=tunnel) as launch, \
                patch.dict(os.environ, {'KAGGLE_TEST':'private','LLAMA_API_KEY':'private','LOCAL_DESK_API_KEY':'private'}):
            self.assertIs(kaggle_server.start_tunnel(server, allow_short_key=True), tunnel)
            self.assertEqual(kaggle_server.tunnel_url(), 'https://desk.ngrok-free.dev/v1')
            command = launch.call_args.args[0]
            environment = launch.call_args.kwargs['env']
            self.assertEqual(command[command.index('--url')+1], 'https://desk.ngrok-free.dev')
            self.assertIn('--inspect=false', command)
            self.assertNotIn(values['NGROK_AUTHTOKEN'], repr(command))
            self.assertEqual(environment['NGROK_AUTHTOKEN'], values['NGROK_AUTHTOKEN'])
            self.assertNotIn('KAGGLE_TEST', environment)
            self.assertNotIn('LLAMA_API_KEY', environment)
            self.assertNotIn('LOCAL_DESK_API_KEY', environment)
            for path in Path(folder).glob('*.json'):
                self.assertNotIn(values['NGROK_AUTHTOKEN'], path.read_text())
                self.assertNotIn(values['LOCAL_DESK_API_KEY'], path.read_text())
            server.terminate.assert_not_called()

    def test_ngrok_missing_setup_prevents_model_download_or_start(self):
        with patch.object(kaggle_server, 'ngrok_domain', side_effect=ValueError('Missing setup')), \
                patch.object(kaggle_server, 'prepare') as prepare, \
                patch.object(kaggle_server, 'start_server') as start:
            with self.assertRaises(ValueError):
                kaggle_server.launch_service(allow_short_key=True)
            prepare.assert_not_called()
            start.assert_not_called()

    def test_latest_tunnel_address_ignores_old_or_invalid_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tunnel.log'
            path.write_text('banner\n{"type":"opened","status":"success","address":"old.lhr.life"}\n', encoding='utf-8')
            offset = path.stat().st_size
            with path.open('a', encoding='utf-8') as output:
                output.write('{"type":"opened","status":"success","address":"new.lhr.life"}\n')
                output.write('{"type":"opened","status":"success","address":"evil.example/path.lhr.life"}\n')
            self.assertEqual(kaggle_server.tunnel_url(path, offset), 'https://new.lhr.life/v1')

    def test_launcher_prepares_then_starts_without_restarting_ready_assets(self):
        server, tunnel = Mock(), Mock()
        tunnel.poll.return_value = None
        with patch.object(kaggle_server.Path, 'is_file', return_value=True), \
                patch.object(kaggle_server.Path, 'exists', return_value=False), \
            patch.object(kaggle_server, 'ngrok_domain', return_value='desk.ngrok-free.dev'), \
            patch.object(kaggle_server, 'read_secret', return_value='test-token'), \
                patch.object(kaggle_server, 'prepare') as prepare, \
                patch.object(kaggle_server, 'start_server', return_value=server), \
                patch.object(kaggle_server, 'start_tunnel', return_value=tunnel), \
                patch.object(kaggle_server, 'tunnel_url', return_value='https://new.lhr.life/v1'):
            self.assertEqual(kaggle_server.launch_service(allow_short_key=True),
                             (server, tunnel, 'https://new.lhr.life/v1'))
            prepare.assert_not_called()

    def test_preflight_requires_two_free_t4_devices(self):
        result = Mock(stdout='0, Tesla T4, 15360, 14912\n1, Tesla T4, 15360, 14912\n')
        with patch.object(kaggle_server.subprocess, 'run', return_value=result), \
                patch.object(kaggle_server.Path, 'mkdir'), \
                patch.object(kaggle_server.shutil, 'disk_usage', return_value=Mock(free=30*1024**3)):
            self.assertEqual(len(kaggle_server.inspect_environment()['devices']), 2)
            for output in ('0, Tesla P100, 16384, 16000\n',
                           '0, Tesla T4, 15360, 13999\n1, Tesla T4, 15360, 14912\n'):
                result.stdout = output
                with self.assertRaises(RuntimeError):
                    kaggle_server.inspect_environment()

    def test_command_keeps_split_mtp_and_ub128(self):
        command = kaggle_server.server_command()
        for flag, value in {'--device':'CUDA0,CUDA1', '--split-mode':'tensor', '--fit':'off',
                            '--spec-type':'draft-mtp', '--spec-draft-n-max':'3',
                            '--ubatch-size':'128', '--host':'127.0.0.1',
                            '--ctx-size':'196608', '--verbosity':'4'}.items():
            self.assertEqual(command[command.index(flag)+1], value)
        for flag in ('--api-key', '--tensor-split', '--spec-draft-model', '--spec-draft-device'):
            self.assertNotIn(flag, command)
        for context in (True, 1024, 196609, '196608'):
            with self.assertRaises(ValueError):
                kaggle_server.server_command(context)

    def test_mmvq_patch_is_idempotent_and_refuses_unknown_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'ggml'/'src'/'ggml-cuda'/'mmvq.cu'
            path.parent.mkdir(parents=True)
            original = ('#include "mmvq.cuh"\n#include <type_traits>\n\n' + kaggle_server.MMVQ_ANCHOR +
                        '    return ne11 <= 8;\n}\n')
            path.write_text(original, encoding='utf-8')
            kaggle_server.patch_mmvq(folder)
            patched = path.read_text(encoding='utf-8')
            self.assertEqual(patched.count('GGML_CUDA_MMVQ_MAX_BATCH_'), 1)
            self.assertEqual(patched.count('#include <array>'), 1)
            self.assertLess(patched.index('t4_override'), patched.index('return ne11 <= 8;'))
            kaggle_server.patch_mmvq(folder)
            self.assertEqual(path.read_text(encoding='utf-8'), patched)
            path.write_text(original.replace('int64_t ne11', 'int64_t n'), encoding='utf-8')
            with self.assertRaises(RuntimeError):
                kaggle_server.patch_mmvq(folder)
            path.write_text(patched.replace('LOCAL-DESK-T4', 'OTHER'), encoding='utf-8')
            with self.assertRaises(RuntimeError):
                kaggle_server.patch_mmvq(folder)

    def test_build_cache_restores_only_matching_intact_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            root, cache = Path(folder)/'root', Path(folder)/'cache'
            binary = root/'llama.cpp/build/bin/llama-server'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'server-v1')
            with patch.object(kaggle_server, 'ROOT', root), patch.object(kaggle_server, 'CACHE', cache):
                kaggle_server.store_build('key-a')
                binary.unlink()
                self.assertFalse(kaggle_server.restore_build('key-b'))
                self.assertFalse(binary.exists())
                self.assertTrue(kaggle_server.restore_build('key-a'))
                self.assertEqual(binary.read_bytes(), b'server-v1')
                (cache/'llama-server').write_bytes(b'server-v2')
                self.assertFalse(kaggle_server.restore_build('key-a'))
                (cache/'build.json').write_text('[]', encoding='utf-8')
                self.assertFalse(kaggle_server.restore_build('key-a'))
        with patch.object(kaggle_server.subprocess, 'run', return_value=Mock(stdout='release 12.8')):
            first = kaggle_server.build_key()
            self.assertEqual(first, kaggle_server.build_key())
        with patch.object(kaggle_server.subprocess, 'run', return_value=Mock(stdout='release 12.9')):
            self.assertNotEqual(first, kaggle_server.build_key())

    def test_prepare_uses_build_cache_or_builds_and_stores(self):
        import hashlib
        import sys
        import types
        names = (kaggle_server.MODEL_FILE, kaggle_server.PROJECTOR_FILE)
        siblings = [Mock(rfilename=name, size=len(name.encode()),
                         lfs=Mock(sha256=hashlib.sha256(name.encode()).hexdigest())) for name in names]
        def download(repo, name, revision, local_dir):
            path = Path(local_dir)/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            return str(path)
        hub = types.SimpleNamespace(hf_hub_download=download, HfApi=Mock(return_value=Mock(
            model_info=Mock(return_value=Mock(sha='revision', siblings=siblings)))))
        for cached in (True, False):
            with self.subTest(cached=cached), tempfile.TemporaryDirectory() as folder, \
                    patch.object(kaggle_server, 'ROOT', Path(folder)), \
                    patch.dict(sys.modules, {'huggingface_hub': hub}), \
                    patch.object(kaggle_server, 'inspect_environment', return_value={'nvcc': '/usr/bin/nvcc'}), \
                    patch.object(kaggle_server, 'build_key', return_value='key'), \
                    patch.object(kaggle_server, 'restore_build', return_value=cached) as restore, \
                    patch.object(kaggle_server, 'store_build') as store, \
                    patch.object(kaggle_server, 'patch_mmvq') as patch_source, \
                    patch.object(kaggle_server.subprocess, 'check_output',
                                 side_effect=[kaggle_server.RUNTIME_TAG, kaggle_server.RUNTIME_COMMIT]), \
                    patch.object(kaggle_server.subprocess, 'run') as run:
                kaggle_server.prepare()
                tools = [call.args[0][0] for call in run.call_args_list]
                restore.assert_called_once_with('key')
                self.assertTrue((Path(folder)/'manifest.json').is_file())
                if cached:
                    self.assertNotIn('git', tools)
                    self.assertNotIn('cmake', tools)
                    store.assert_not_called()
                    patch_source.assert_not_called()
                else:
                    self.assertIn('git', tools)
                    self.assertEqual(tools.count('cmake'), 2)
                    store.assert_called_once_with('key')
                    patch_source.assert_called_once()

    def test_server_environment_sets_mmvq_threshold(self):
        process = Mock(pid=77)
        process.poll.return_value = None
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(kaggle_server, 'ROOT', Path(folder)), \
                patch.object(kaggle_server, 'read_secret', return_value='test-key'), \
                patch.object(kaggle_server.socket, 'socket'), \
                patch.object(kaggle_server.subprocess, 'Popen', return_value=process) as launch, \
                patch.object(kaggle_server.urllib.request, 'urlopen', return_value=response), \
                patch.object(kaggle_server.json, 'load', return_value={'status':'ok'}):
            (Path(folder)/'manifest.json').write_text('{}', encoding='utf-8')
            self.assertIs(kaggle_server.start_server(allow_short_key=True), process)
        environment = launch.call_args.kwargs['env']
        self.assertEqual(environment['GGML_CUDA_MMVQ_MAX_BATCH'], '1')
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '0,1')

    def test_short_key_requires_explicit_approval(self):
        with patch.dict(os.environ, {'LOCAL_DESK_API_KEY':'test-short-key'}):
            with self.assertRaises(ValueError):
                kaggle_server.read_secret('LOCAL_DESK_API_KEY')
            self.assertEqual(kaggle_server.read_secret('LOCAL_DESK_API_KEY', allow_short=True),
                             'test-short-key')
        with patch.dict(os.environ, {'LOCAL_DESK_API_KEY':'invalid key'}):
            with self.assertRaises(ValueError):
                kaggle_server.read_secret('LOCAL_DESK_API_KEY', allow_short=True)

    def test_existing_server_prevents_launch(self):
        with patch.object(kaggle_server.Path, 'is_file', return_value=True), \
                patch.object(kaggle_server, 'read_secret', return_value='test-key'), \
                patch.object(kaggle_server.socket, 'socket') as socket_factory, \
                patch.object(kaggle_server.subprocess, 'Popen') as launch:
            socket_factory.return_value.__enter__.return_value.bind.side_effect = OSError('Port in use')
            with self.assertRaises(OSError):
                kaggle_server.start_server(allow_short_key=True)
            socket_factory.return_value.__enter__.return_value.setsockopt.assert_called_once_with(
                kaggle_server.socket.SOL_SOCKET, kaggle_server.socket.SO_REUSEADDR, 1)
            launch.assert_not_called()

    def test_tunnel_refuses_server_without_authentication(self):
        server = Mock()
        server.poll.return_value = None
        with patch.object(kaggle_server, 'read_secret', return_value='test-key'), \
                patch.object(kaggle_server.urllib.request, 'urlopen') as urlopen, \
                patch.object(kaggle_server.subprocess, 'Popen') as launch:
            urlopen.return_value.__enter__.return_value = Mock()
            with self.assertRaisesRegex(RuntimeError, 'invalid API key'):
                kaggle_server.start_tunnel(server, allow_short_key=True)
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()
