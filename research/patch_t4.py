"""Research-only llama.cpp patches for the T4 study (idempotent). Usage: python patch_t4.py <src-root>"""
import sys
from pathlib import Path

MARK = '// T4-STUDY'

MMVQ_ANCHOR = ('bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {\n'
               '    if (!ggml_is_quantized(type)) {\n'
               '        return false;\n'
               '    }\n')
MMVQ_OVERRIDE = (
    '    ' + MARK + ': optional env override of the MMVQ/MMQ batch threshold; unset = upstream behaviour\n'
    '    static const std::array<int, GGML_TYPE_COUNT> t4_override = [] {\n'
    '        std::array<int, GGML_TYPE_COUNT> table;\n'
    '        const char * fallback = getenv("GGML_CUDA_MMVQ_MAX_BATCH");\n'
    '        const int base = fallback ? atoi(fallback) : -1;\n'
    '        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {\n'
    '            const char * name = ggml_type_name((ggml_type) i);\n'
    '            const std::string key = std::string("GGML_CUDA_MMVQ_MAX_BATCH_") + (name ? name : "");\n'
    '            const char * value = getenv(key.c_str());\n'
    '            table[i] = value ? atoi(value) : base;\n'
    '        }\n'
    '        return table;\n'
    '    }();\n'
    '    if (t4_override[type] >= 0) {\n'
    '        return ne11 <= t4_override[type];\n'
    '    }\n')
INCLUDE_ANCHOR = '#include <type_traits>\n'
INCLUDES = '#include <type_traits>\n#include <array>\n#include <cstdlib>\n#include <string>\n'

PERF_ANCHOR = ('static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {\n'
               '    std::vector<std::unique_ptr<test_case>> test_cases;\n')
PERF_SHAPES = (
    '    ' + MARK + ': Qwen3.8-27B decode shapes (m = output rows, k = input width)\n'
    '    for (ggml_type type_a : {GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,\n'
    '                             GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {\n'
    '        for (int64_t n : {1, 4}) {\n'
    '            for (auto mk : std::vector<std::pair<int64_t, int64_t>>{{17408, 5120}, {5120, 17408}, {10240, 5120},\n'
    '                    {6144, 5120}, {5120, 6144}, {12288, 5120}}) {\n'
    '                test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, mk.first, n, mk.second, {1, 1}, {1, 1}));\n'
    '            }\n'
    '        }\n'
    '    }\n'
    '    for (ggml_type type_a : {GGML_TYPE_Q3_K, GGML_TYPE_Q4_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {\n'
    '        for (int64_t n : {1, 4}) {\n'
    '            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 62080, n, 5120, {1, 1}, {1, 1}));\n'
    '        }\n'
    '    }\n'
    '    ' + MARK + ' end\n\n')


def strip_old_perf(path):
    text = path.read_text()
    start = text.find('    ' + MARK + ': Qwen3.8-27B decode shapes')
    if start < 0 or PERF_SHAPES in text:
        return 'none'
    end_marker = '    ' + MARK + ' end\n\n'
    end = text.find(end_marker, start)
    end = end+len(end_marker) if end >= 0 else text.index('\n    }\n\n', start)+len('\n    }\n\n')
    path.write_text(text[:start]+text[end:])
    return 'removed'


def patch(path, anchor, replacement):
    text = path.read_text()
    if replacement in text:
        return 'already'
    if text.count(anchor) != 1:
        raise SystemExit('anchor count %d in %s' % (text.count(anchor), path))
    path.write_text(text.replace(anchor, replacement))
    return 'patched'


def main(root):
    root = Path(root)
    mmvq = root / 'ggml' / 'src' / 'ggml-cuda' / 'mmvq.cu'
    print('mmvq include', patch(mmvq, INCLUDE_ANCHOR, INCLUDES))
    print('mmvq override', patch(mmvq, MMVQ_ANCHOR, MMVQ_ANCHOR+MMVQ_OVERRIDE))
    tests = root / 'tests' / 'test-backend-ops.cpp'
    print('old perf shapes', strip_old_perf(tests))
    print('perf shapes', patch(tests, PERF_ANCHOR, PERF_ANCHOR+PERF_SHAPES))


if __name__ == '__main__':
    main(sys.argv[1])
