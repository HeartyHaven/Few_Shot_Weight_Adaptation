"""Probe an existing torch installation without loading weights or writing caches."""
import datetime
import importlib.metadata
import json
import sys
import warnings

result = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'python': sys.executable, 'weights_loaded': False,
          'peak_memory_measured': False}
try:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        import torch
        result.update(torch_version=torch.__version__, compiled_cuda=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(),
                      cuda_device_count=torch.cuda.device_count())
        result['warnings'] = [str(item.message) for item in captured]
    result['transformers_version'] = importlib.metadata.version('transformers')
except Exception as exc:
    result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
print(json.dumps(result, indent=2, ensure_ascii=False))
