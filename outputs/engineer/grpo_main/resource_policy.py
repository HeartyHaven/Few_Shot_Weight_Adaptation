"""Admission budgets only; scientific training/evaluation parameters stay in their workers."""
import math

POLICY_VERSION = 2
HEADROOM_GIB = 8
MAX_WORKERS = 3
MAX_TRAINERS = 2
ENV_PROFILE = 'FSWA_RESOURCE_PROFILE'
ENV_RESERVE = 'FSWA_RESERVED_GIB'
ENV_CAP = 'FSWA_TORCH_CAP_GIB'


def legacy_reserve(job):
    if job['kind'] == 'evaluate':
        return 72 if job['model'] == 'Qwen3-14B' else 44
    budgets = ({'Qwen3-1.7B': 48, 'Qwen3-4B': 64, 'Qwen3-14B': 100}
               if job['baseline'] == 'distill_k' else
               {'Qwen3-1.7B': 75, 'Qwen3-4B': 90, 'Qwen3-14B': 128})
    return budgets[job['model']]


def profiles(job):
    if job['kind'] == 'evaluate':
        return [dict(profile='eval-fixed-kv', reserved_gib=legacy_reserve(job), torch_cap_gib=None,
                     eval_mem=job.get('eval_mem', .50 if job['model'] == 'Qwen3-14B' else .28))]
    mb = job['microbatch']
    if job['baseline'] == 'distill_k':
        reserve = legacy_reserve(job)
        return [dict(profile='sft-capped', reserved_gib=reserve, torch_cap_gib=reserve-4, microbatch=mb)]
    if job.get('oom'):
        reserve = legacy_reserve(job)
        return [dict(profile='grpo-oom-recovery', reserved_gib=reserve, torch_cap_gib=reserve-6,
                     microbatch=mb)]
    reserve, cap = {'Qwen3-1.7B': (64, 58), 'Qwen3-4B': (66, 60), 'Qwen3-14B': (128, 122)}[job['model']]
    result = [dict(profile='grpo-capped', reserved_gib=reserve, torch_cap_gib=cap, microbatch=mb)]
    # A smaller microbatch retains the same generation and optimizer batch through accumulation.
    if job['model'] == 'Qwen3-1.7B':
        result.append(dict(profile='grpo-compact', reserved_gib=56, torch_cap_gib=50, microbatch=min(mb, 4)))
    return result


def recovered_profile(job, env):
    """Never retroactively shrink the reservation of an old, uncapped process."""
    if ENV_RESERVE not in env:
        return dict(resource_profile='legacy', reserved_gib=legacy_reserve(job), torch_cap_gib=None)
    reserve = float(env[ENV_RESERVE])
    cap = float(env[ENV_CAP]) if env.get(ENV_CAP) else None
    if not math.isfinite(reserve) or not 0 < reserve <= 128:
        raise ValueError('Invalid recovered memory reservation')
    if cap is not None and (not math.isfinite(cap) or not 0 < cap <= reserve-4):
        raise ValueError('Invalid recovered allocator cap')
    return dict(resource_profile=env.get(ENV_PROFILE, 'recovered'), reserved_gib=reserve, torch_cap_gib=cap)


def committed_gib(active, snapshot):
    observed = snapshot.get('job_used_gib', {})
    committed = snapshot.get('foreign_gib', 0) + sum(
        max(j.get('reserved_gib', legacy_reserve(j)), observed.get(j['id'], 0)) for j in active)
    return max(snapshot['used_gib'], committed)


def fits(job, profile, active, snapshot):
    if len(active) >= MAX_WORKERS:
        return False
    if any(j.get('oom') for j in active) or (job.get('oom') and active):
        return False  # Retry an OOM alone, without sacrificing another live worker.
    if job['kind'] == 'train' and sum(j['kind'] == 'train' for j in active) >= MAX_TRAINERS:
        return False
    return committed_gib(active, snapshot) + profile['reserved_gib'] <= snapshot['total_gib'] - HEADROOM_GIB


def choose_profile(job, active, snapshot):
    return next((p for p in profiles(job) if fits(job, p, active, snapshot)), None)
