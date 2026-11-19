"""Record explicitly missing WP0 measurements after failed preflight, never zeros."""
import json
from pathlib import Path

out = Path(__file__).resolve().parent.parent
preflight = json.loads((out / 'preflight.json').read_text())
torch_probe = json.loads((out / 'torch_probe.json').read_text())
assert torch_probe.get('cuda_available') is False, 'Reassess: GPU is now available'
assert not preflight['dns']['huggingface.co']['resolved'], 'Reassess: HF DNS now works'
reason = 'This execution session has no accessible CUDA devices and cannot resolve Hugging Face/PyPI.'
models = ['google/gemma-3-1b-it', 'Qwen/Qwen3-1.7B']
tasks = ['medcalc_verified_rule_based'] + [
    f'tracking_shuffled_objects_{n}_objects' for n in ['three', 'five', 'seven']]


def save(name, value):
    # Refuse to replace any potentially real measurements on a later rerun.
    with (out / name).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


common = {'status': 'blocked', 'reason': reason,
          'evidence': ['preflight.json', 'torch_probe.json'],
          'measurements_obtained': False}
save('baseline.json', {
    **common,
    'conditions': [
        {'model': model, 'task': task, 'condition': condition, 'answer_mode': mode,
         'max_new_tokens': limit, 'status': 'not_run', 'n': None, 'correct': None,
         'accuracy': None, 'bootstrap_95_ci': None, 'raw_output_file': None}
        for model in models for task in tasks for condition in ['B0', 'B1']
        for mode, limit in [('direct', 16), ('free', 512)]],
    'admission': [{'model': model, 'task': task, 'status': 'not_evaluated',
                   'baseline_gate': None, 'prompt_improvement_gate': None}
                  for model in models for task in tasks],
    'medcalc_inventory': {'status': 'not_run', 'rule_based_calculator_count': None,
                         'test_counts_by_calculator': None, 'teaching_group_candidates': None},
    'deterministic_generation': {model: {'status': 'not_run', 'bitwise_identical': None}
                                 for model in models},
    'test_items_read': False, 'prompts_written_or_tuned': False
})
save('memory.json', {**common, 'planned_batch_size': 1, 'planned_sequence_length': 1024,
                    'measurements': [
                        {'model': model, 'mode': mode, 'status': 'not_run',
                         'max_memory_allocated_bytes': None, 'ratio_to_inference': None}
                        for model in models
                        for mode in ['inference', 'all_layer_activation_hooks', 'single_layer_target_backward']]})
save('smoke_edit.json', {**common,
                        'models': {model: {'status': 'not_run', 'module_structure_verified': False,
                                           'has_post_mlp_norm': None, 'rank_one_edit_applied': False,
                                           'downstream_delta_measured': False,
                                           'normalization_attenuation_conclusion': None}
                                   for model in models}})
with (out / 'gemma_arch.txt').open('x') as f:
    f.write('NOT MEASURED. Gemma weights are not cached at either checked HF_HOME location,\n'
            'Hugging Face DNS fails, and this session has no accessible CUDA device.\n'
            'No Gemma model was loaded or printed. No architecture or normalization\n'
            'claim has been inferred from a different Gemma model.\n'
            'Evidence: preflight.json; torch_probe.json.\n')
print(json.dumps({'status': 'blocked', 'output_directory': str(out)}, ensure_ascii=False))
