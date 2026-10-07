"""Rewrite one tensor of a GGUF file with another quantization type (T4 study, research only).
Usage: python requant_head.py <src.gguf> <dst.gguf> <tensor-name> <GGML type, e.g. Q4_0>"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, next(str(path) for path in (Path('/kaggle/temp/t4-engine/src/b11000/gguf-py'),
                                               Path('/kaggle/temp/t4-engine/src/v050/gguf-py')) if path.exists()))
import gguf
from gguf import quants

CHUNK_ROWS = 8192


def requantize(tensor, target):
    rows = tensor.data.shape[0]
    parts = []
    for start in range(0, rows, CHUNK_ROWS):
        values = quants.dequantize(tensor.data[start:start+CHUNK_ROWS], tensor.tensor_type)
        parts.append(quants.quantize(values.astype(np.float32), target))
    return np.concatenate(parts, axis=0)


def main(src, dst, name, type_name):
    target = gguf.GGMLQuantizationType[type_name]
    reader = gguf.GGUFReader(src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)
    replaced = {}
    for tensor in reader.tensors:
        if tensor.name == name:
            data = requantize(tensor, target)
            replaced[name] = data
            writer.add_tensor_info(tensor.name, data.shape, data.dtype, data.nbytes, target)
            print('replaced', name, tensor.tensor_type.name, '->', target.name, tensor.n_bytes, '->', data.nbytes, flush=True)
        else:
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)
    if name not in replaced:
        raise SystemExit('tensor not found: '+name)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(replaced.get(tensor.name, tensor.data), tensor_endianess=reader.endianess)
    writer.close()
    print('written', dst, Path(dst).stat().st_size, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
