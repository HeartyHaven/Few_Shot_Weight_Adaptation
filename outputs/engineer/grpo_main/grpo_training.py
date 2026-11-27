"""Pinned TRL GRPO, verifiable rewards, LoRA, and a frozen reference policy."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

from common import HERE, ROOT, atomic_json, model_path, progress, read_json, run_name, sha, dataset_dir

PROTOCOL = {
    'algorithm':'GRPO', 'implementation':'trl==0.25.1', 'loss_type':'grpo',
    'epochs':1, 'num_generations':8, 'prompts_per_update':8, 'completions_per_update':64,
    'learning_rate':1e-5, 'beta':0.04, 'epsilon':0.2, 'num_iterations':1,
    'scale_rewards':'group', 'reward':'correct final answer=1; otherwise=0; existing task grader',
    'max_prompt_length':4096, 'max_completion_length':2048,
    'temperature':1.0, 'top_p':1.0, 'top_k':-1,
    'rank':16, 'alpha':32, 'dropout':0.0, 'dtype':'bfloat16',
    'mask_truncated_completions':True, 'vllm_importance_sampling_correction':True,
    'vllm_importance_sampling_cap':2.0, 'reference':'original base model, adapters disabled',
    'selection':'final fixed-budget checkpoint; no hidden validation/test selection',
}


def reward_values(completions, reward_item):
    from tdwa import grade as G
    values=[]
    for completion, payload in zip(completions,reward_item,strict=True):
        text=completion if isinstance(completion,str) else ''.join(m['content'] for m in completion)
        values.append(float(G.grade(json.loads(payload),text)['correct']))
    return values


def train(a):
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    os.environ['WANDB_DISABLED']='true'
    # TRL's four preferred ports race when independent single-GPU jobs start together.
    # Ask the OS for an ephemeral localhost port before constructing Accelerate/vLLM.
    import socket
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as master_socket:
        master_socket.bind(('127.0.0.1',0))
        os.environ['MASTER_ADDR']='127.0.0.1'
        os.environ['MASTER_PORT']=str(master_socket.getsockname()[1])

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer
    import trl.trainer.grpo_trainer as trl_module
    from tdwa import grade as G

    torch.set_num_threads(4)
    # Actor and colocated vLLM use this same process allocator. External CUDA overhead
    # has a separate reservation margin; this is not a cap on evaluator subprocesses.
    cap_gib=float(os.environ['FSWA_TORCH_CAP_GIB']) if os.environ.get('FSWA_TORCH_CAP_GIB') else None
    reserve_gib=float(os.environ['FSWA_RESERVED_GIB']) if os.environ.get('FSWA_RESERVED_GIB') else None
    if cap_gib is not None:
        if not reserve_gib or not 0 < cap_gib <= reserve_gib-4:
            raise RuntimeError('Invalid scheduler allocator budget')
        torch.cuda.set_per_process_memory_fraction(cap_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
    name=run_name(a.task,a.model,a.seed,a.baseline)
    out=HERE/('pilots' if a.smoke else 'runs')/name
    out.mkdir(parents=True,exist_ok=True)
    beat=out/'train_progress.json'
    started=time.time()
    progress(beat,phase='loading_data')
    ready=read_json(dataset_dir(a.baseline)/f'{a.task}.ready.json')
    data_path=dataset_dir(a.baseline)/f'{a.task}.jsonl'
    if not ready or sha(data_path)!=ready['sha256']:
        raise RuntimeError('Frozen GRPO training data missing or changed')
    rows=[json.loads(x) for x in data_path.read_text().splitlines()]
    prompts_per_update=4 if a.baseline=='grpo_100' else 8
    generation_batch=8*prompts_per_update
    protocol={**PROTOCOL,'prompts_per_update':prompts_per_update,'completions_per_update':generation_batch}
    assert len(rows)==ready['n_examples'] and len(rows)>=100 and len(rows)%prompts_per_update==0
    tok=AutoTokenizer.from_pretrained(model_path(a.model),local_files_only=True,padding_side='left')
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    dataset=[]
    for row in rows:
        prompt=tok.apply_chat_template([{'role':'user','content':G.format_prompt(row)}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if len(tok.encode(prompt,add_special_tokens=False))>PROTOCOL['max_prompt_length']:
            raise RuntimeError('Unexpected prompt truncation; stop instead of changing the frozen data')
        item={k:row[k] for k in ['id','task','answer','lower','upper'] if k in row}
        dataset.append({'prompt':prompt,'reward_item':json.dumps(item,ensure_ascii=False)})
    expected_steps=a.max_steps if a.smoke else len(rows)//prompts_per_update
    assert a.microbatch in [1,2,4,8] and generation_batch%a.microbatch==0
    config_record={'protocol':protocol,'baseline':a.baseline,'task':a.task,'student':'Qwen/'+a.model,'seed':a.seed,
                   'n_examples':len(rows),'expected_steps':expected_steps,'data_sha256':ready['sha256'],
                   'grader_sha256':sha(ROOT/'tdwa/grade.py'),'microbatch':a.microbatch,'smoke':a.smoke,
                   'training_source_sha256':sha(Path(__file__)),
                   'resource_profile':os.environ.get('FSWA_RESOURCE_PROFILE','legacy'),
                   'scheduler_reserved_gib':reserve_gib,'torch_cap_gib':cap_gib}
    old=read_json(out/'config.json')
    if old and any(old[k]!=config_record[k] for k in ['protocol','task','student','seed','data_sha256','expected_steps']):
        raise RuntimeError('Refusing to resume a different GRPO protocol')
    atomic_json(out/'config.json',config_record)
    stats=read_json(out/'rollout_stats.json',{'batches':0,'sequences':0,'correct':0,'unique_ids':[]})
    seen=set(stats['unique_ids'])

    def correctness_reward(completions,reward_item,**kwargs):
        values=reward_values(completions,reward_item)
        ids=[json.loads(x)['id'] for x in reward_item]
        assert len(ids)%8==0
        for i in range(0,len(ids),8):
            assert len(set(ids[i:i+8]))==1,'Incorrect GRPO prompt grouping'
        seen.update(ids)
        stats.update(batches=stats['batches']+1,sequences=stats['sequences']+len(values),
                     correct=stats['correct']+sum(values),unique_ids=sorted(seen))
        atomic_json(out/'rollout_stats.json',stats)
        record={'time':time.time(),'ids':ids,'rewards':values,'completions':completions}
        with gzip.open(out/'rollouts.jsonl.gz','at',encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
        progress(beat,phase='reward',rollout_batches=stats['batches'],unique_prompts=len(seen))
        return values

    # Fixed KV allocations avoid dependence on allocations made by other processes on the GPU.
    original_llm=trl_module.LLM
    def fixed_llm(*args,**kwargs):
        kwargs.update(kv_cache_memory_bytes=(24 if a.model=='Qwen3-14B' else 16)*2**30,
                      dtype='bfloat16',seed=a.seed,max_num_batched_tokens=8192)
        return original_llm(*args,**kwargs)
    trl_module.LLM=fixed_llm

    class FrozenBaseGRPOTrainer(GRPOTrainer):
        """Copy merged weights to vLLM without in-place bf16 merge/unmerge drift of the reference."""
        sync_count=0
        nonzero_sync_count=0
        def _move_model_to_vllm(self):
            if self.accelerator.num_processes!=1 or self.vllm_mode!='colocate':
                raise RuntimeError('This validated worker uses one GPU and colocated vLLM')
            model=self.accelerator.unwrap_model(self.model)
            llm_model=self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            had_delta=False
            progress(beat,phase='sync_rollout_weights',step=self.state.global_step)
            with torch.no_grad():
                for name,param in model.named_parameters():
                    if '.lora_' in name or 'original_module' in name:
                        continue
                    target=name.removeprefix('base_model.model.').replace('.base_layer','')
                    target=self._fix_param_name_to_vllm(target,extra_prefixes=['modules_to_save.default.'])
                    if name.endswith('.base_layer.weight'):
                        layer=model.get_submodule(name.removesuffix('.base_layer.weight'))
                        merged=param.detach().float()
                        for adapter in layer.active_adapters:
                            if adapter in layer.lora_A:
                                if not had_delta: had_delta=bool(torch.count_nonzero(layer.lora_B[adapter].weight))
                                merged=merged+layer.get_delta_weight(adapter).float()
                        weight=merged.to(param.dtype)
                        llm_model.load_weights([(target,weight)])
                        del merged,weight
                    else:
                        llm_model.load_weights([(target,param.detach())])
            self.llm.reset_prefix_cache()
            self.sync_count+=1
            self.nonzero_sync_count+=int(had_delta)

    class Monitor(TrainerCallback):
        def on_step_begin(self,args,state,control,**kwargs):
            progress(beat,phase='training',step=state.global_step,expected_steps=expected_steps)
        def on_substep_end(self,args,state,control,**kwargs):
            progress(beat,phase='gradient_accumulation',step=state.global_step,expected_steps=expected_steps)
        def on_step_end(self,args,state,control,**kwargs):
            progress(beat,phase='training',step=state.global_step,expected_steps=expected_steps)
        def on_save(self,args,state,control,**kwargs):
            progress(beat,phase='checkpoint_saved',step=state.global_step,expected_steps=expected_steps)
        def on_log(self,args,state,control,logs=None,**kwargs):
            for key,value in (logs or {}).items():
                if isinstance(value,(int,float)) and not math.isfinite(value):
                    raise RuntimeError(f'Nonfinite training metric: {key}')
            row={'time':time.time(),'step':state.global_step,'wall_seconds':time.time()-started,
                 'allocated_gib':torch.cuda.memory_allocated()/2**30,
                 'reserved_gib':torch.cuda.memory_reserved()/2**30,
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,**(logs or {})}
            with (out/'train_log.jsonl').open('a') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')

    args=GRPOConfig(
        output_dir=str(out/'checkpoints'),seed=a.seed,data_seed=a.seed,
        model_init_kwargs={'dtype':torch.bfloat16,'attn_implementation':'sdpa','local_files_only':True},
        per_device_train_batch_size=a.microbatch,gradient_accumulation_steps=generation_batch//a.microbatch,
        per_device_eval_batch_size=8,generation_batch_size=generation_batch,num_generations=8,
        num_train_epochs=1,max_steps=a.max_steps if a.smoke else -1,
        learning_rate=1e-5,beta=0.04,epsilon=0.2,num_iterations=1,
        loss_type='grpo',scale_rewards='group',max_prompt_length=4096,max_completion_length=2048,
        temperature=1.0,top_p=1.0,top_k=-1,
        bf16=True,fp16=False,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        optim='adamw_torch_fused',weight_decay=0.0,lr_scheduler_type='cosine',warmup_ratio=0.03,
        max_grad_norm=1.0,disable_dropout=True,mask_truncated_completions=True,
        use_vllm=True,vllm_mode='colocate',vllm_gpu_memory_utilization=0.30,
        vllm_tensor_parallel_size=1,vllm_enable_sleep_mode=False,
        vllm_importance_sampling_correction=True,vllm_importance_sampling_cap=2.0,
        logging_steps=1,save_strategy='steps',save_steps=25,save_total_limit=2,
        report_to='none',disable_tqdm=True,remove_unused_columns=False,
        dataloader_num_workers=0,dataloader_drop_last=False,
    )
    peft=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,bias='none',task_type='CAUSAL_LM',
                   target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    progress(beat,phase='loading_models')
    trainer=FrozenBaseGRPOTrainer(model=str(model_path(a.model)),args=args,peft_config=peft,
             reward_funcs=correctness_reward,train_dataset=Dataset.from_list(dataset),processing_class=tok,
             callbacks=[Monitor()])

    def base_fingerprint():
        digest=hashlib.sha256()
        with torch.no_grad():
            for name,param in trainer.model.named_parameters():
                if not param.requires_grad:
                    values=param.detach().flatten()[::max(1,param.numel()//32)][:32]
                    digest.update(name.encode());digest.update(values.float().cpu().numpy().tobytes())
        return digest.hexdigest()
    base_before=base_fingerprint()
    ck=get_last_checkpoint(str(out/'checkpoints')) if (out/'checkpoints').exists() else None
    progress(beat,phase='training_start',checkpoint=ck,expected_steps=expected_steps)
    result=trainer.train(resume_from_checkpoint=ck)
    if trainer.state.global_step!=expected_steps:
        raise RuntimeError(f'Incomplete optimization: {trainer.state.global_step}/{expected_steps}')
    if base_fingerprint()!=base_before:
        raise RuntimeError('Frozen base weights changed during GRPO')
    if not a.smoke and len(seen)!=len(rows):
        raise RuntimeError(f'Training coverage incomplete: {len(seen)}/{len(rows)}')
    adapter=out/'adapter';trainer.save_model(str(adapter))
    trainer.save_state()
    nonzero=sum(int(torch.count_nonzero(v)) for k,v in trainer.model.named_parameters() if '.lora_B.' in k)
    if a.smoke and (nonzero==0 or trainer.nonzero_sync_count==0):
        raise RuntimeError('Pilot did not validate an updated adapter in subsequent rollouts')
    done={**config_record,'name':name,'steps':trainer.state.global_step,'unique_prompts_seen':len(seen),
          'rollout_sequences_including_retries':stats['sequences'],'rollout_syncs':trainer.sync_count,
          'nonzero_rollout_syncs':trainer.nonzero_sync_count,
          'base_unchanged':True,'base_sample_fingerprint':base_before,'nonzero_lora_B':nonzero,
          'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'wall_seconds':time.time()-started,
          'peak_mem_gib':torch.cuda.max_memory_allocated()/2**30,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
          'metrics':result.metrics,'finished':time.time()}
    atomic_json(out/'train_done.json',done)
    progress(beat,phase='complete',step=trainer.state.global_step)
    print('GRPO_TRAINING_COMPLETE',name,json.dumps(result.metrics),flush=True)
