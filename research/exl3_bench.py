"""ExLlamaV3 speed probe for the T4 engine study (Kaggle side)."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from exllamav3 import model_init, Generator
from exllamav3.generator.sampler import ComboSampler

ROWS = '\n'.join('ROW%02d:READY' % index for index in range(1, 25))
PROMPTS = {
    'copy': 'Return exactly the following records, preserving every line and adding no other text:\n'+ROWS,
    'arithmetic': ('There are 7 boxes with 18 units each. Then 29 units are removed and 14 units '
                   'are added. Return only the final total as text.'),
}


def chat(text):
    return '<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n'


def gpu_used():
    output = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True).stdout.split()
    return [int(value) for value in output]


def run(generator, sampler, prompt, max_new, stops):
    text, last = generator.generate(prompt=prompt, max_new_tokens=max_new, sampler=sampler,
                                    completion_only=True, return_last_results=True,
                                    stop_conditions=stops, add_bos=False)
    new = last.get('new_tokens', 0)
    prefill = last.get('time_prefill', 0.0)
    generate = last.get('time_generate', 0.0)
    return {'new_tokens': new, 'time_prefill': prefill, 'time_generate': generate,
            'decode_tps': (new-1)/generate if new > 1 and generate > 0 else None,
            'accepted': last.get('accepted_draft_tokens'), 'rejected': last.get('rejected_draft_tokens'),
            'eos_reason': last.get('eos_reason'), 'text_head': text[:200]}


def main():
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache=True, add_draft_model_args=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max_new', type=int, default=320)
    parser.add_argument('--prefill_repeat', type=int, default=120)
    args = parser.parse_args()
    started = time.time()
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    result = {'args': vars(args), 'load_seconds': time.time()-started, 'gpu_used_after_load': gpu_used()}
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model,
                          draft_cache=draft_cache, num_draft_tokens=args.num_draft_tokens)
    sampler = ComboSampler(temperature=1.0, top_k=1)
    stops = [token for token in (getattr(tokenizer, 'eos_token_id', None),) if token is not None]
    stops.append('<|im_end|>')
    run(generator, sampler, chat('Say hello in one word.'), 16, stops)
    result['warm_cases'] = {name: run(generator, sampler, chat(text), args.max_new, stops)
                            for name, text in PROMPTS.items()}
    result['cases'] = {name: run(generator, sampler, chat(text), args.max_new, stops)
                       for name, text in PROMPTS.items()}

    def filler(offset):
        return chat('Summarize in one word:\n'+' '.join(
            'The quick brown fox jumps over the lazy dog number %d.' % (offset+index)
            for index in range(args.prefill_repeat)))
    result['warm_prefill'] = run(generator, sampler, filler(5000), 2, stops)
    long_prompt = filler(0)
    result['prefill_prompt_tokens'] = int(tokenizer.encode(long_prompt).shape[-1])
    result['prefill'] = run(generator, sampler, long_prompt, 2, stops)
    result['gpu_used_end'] = gpu_used()
    Path(args.out).write_text(json.dumps(result, indent=2))
    print('EXL3_RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
