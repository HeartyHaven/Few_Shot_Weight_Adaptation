"""Baseline B5: prompt baking (Bhargava et al. 2024, arXiv 2409.13697) of the B1 procedure prompt into a LoRA.

  1) sample : the base student WITH the B1 procedure in the prompt answers each training question (vLLM, T=0.8, 4 samples,
              768 new tokens).  Questions: the 100 dev items + 300 self-generated items (generator seed 9393; answers unused),
              i.e. the same data allowance as a teacher (100 real + <=300 own).
  2) train  : LoRA student WITHOUT the prompt is trained to match, token by token over the sampled replies, the full
              next-token distribution of the prompted base model:  KL(p_base(.|procedure + q, y<t) || p_lora(.|q, y<t)).
              The prompted distribution comes from the same model with the adapter disabled.
Same LoRA settings as tdwa/baseline_lora.py (r=16, alpha 32, lr 2e-4, cosine), 2 epochs, batch 8 (1 x 8 accumulation).
Usage: python -m tdwa.baseline_bake sample|train --task t2 --student google/gemma-3-1b-it [--seed 0]
Evaluate: python -m tdwa.evaluate_vllm --task t2 --student <repo> --lora baselines/B5__t2__<tag>__s0/adapter --name B5__t2__<tag>__s0
"""
import argparse, json, math, os, random, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tdwa import grade as G  # noqa: E402
from tdwa import data as D  # noqa: E402

# B1 procedure text, verbatim from pilot/scripts/baseline.py (WP0)
T2_PROC = ("Procedure to follow:\n1. Write down the starting assignment as a table (person -> item).\n"
           "2. Process the swaps strictly in the order given, one at a time. For a swap between X and Y, exchange the "
           "two entries of X and Y and leave everyone else unchanged. After each swap, rewrite the whole table.\n"
           "3. Only after the last swap, read off the entry for the person asked about.\n"
           "4. Pick the option that matches that entry.\n\n")
PROC = {"t2": T2_PROC}


def t1_rules():
    """B1 for MedCalc, as in WP0 (pilot/scripts/baseline.py): the generic criteria list cut from each calculator's
    one-shot worked explanation in MedCalc-Bench-Verified, plus the absent-if-unmentioned convention."""
    import re
    from datasets import load_dataset
    def extract_rule(e):
        keep, seen = [], False
        for line in e.strip().split("\n"):
            numbered = bool(re.match(r"\s*\d+\.\s", line)); sub = bool(re.match(r"\s+\S|\s*[-*•]", line))
            if seen and line.strip() and not numbered and not sub:
                if re.search(r"sum|calculat|total", line, re.I):
                    keep.append(line)
                break
            seen = seen or numbered; keep.append(line)
        return "\n".join(keep).strip()
    return {str(r["Calculator ID"]): extract_rule(r["Ground Truth Explanation"]) for r in load_dataset("nsk7153/MedCalc-Bench-Verified")["one_shot"]}


def prefix_fn(task):
    """item -> text put before the question for condition B1."""
    if task == "t1":
        R = t1_rules()
        return lambda it: ("Scoring rule for this calculator:\n" + R[str(it["calc_id"])] +
                           "\nIf an item is not mentioned in the patient note, assume it is absent.\n\n") if str(it.get("calc_id")) in R else ""
    return lambda it: PROC[task]


def questions(task):
    dev = [json.loads(l) for l in open(os.path.join(ROOT, "data", task, "dev.jsonl"))]
    rng = random.Random(9393); own = []
    if task != "t2":                      # t1: only the 100 dev cases (no generator for new patient notes)
        return dev
    for n, c in ((3, 60), (5, 120), (7, 120)):
        own += [D.make_item(rng, n, None, f"own-{n}-{i}") for i in range(c)]
    return dev + own


def render(tok, repo, user):
    kw = {"enable_thinking": False} if "qwen" in repo.lower() else {}
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["sample", "train"]); ap.add_argument("--task", default="t2"); ap.add_argument("--student", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--k", type=int, default=4); ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--r", type=int, default=16); ap.add_argument("--mem", type=float, default=0.35)
    a = ap.parse_args()
    tag = a.student.split("/")[-1]; out = os.path.join(ROOT, "baselines", f"B5__{a.task}__{tag}__s{a.seed}"); os.makedirs(out, exist_ok=True)
    qs = questions(a.task)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.student)

    if a.phase == "sample":
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        llm = LLM(model=a.student, dtype="bfloat16", gpu_memory_utilization=a.mem, max_model_len=4096, seed=a.seed)
        pf = prefix_fn(a.task)
        ps = [TokensPrompt(prompt_token_ids=tok.encode(render(tok, a.student, pf(q) + G.format_prompt(q)), add_special_tokens=False)) for q in qs]
        outs = llm.generate(ps, SamplingParams(n=a.k, temperature=0.8, max_tokens=768, seed=a.seed))
        n_ok = 0
        with open(os.path.join(out, "samples.jsonl"), "w") as f:
            for q, o in zip(qs, outs):
                for c in o.outputs:
                    g = G.grade(q, c.text); n_ok += g["correct"]
                    f.write(json.dumps({"id": q["id"], "reply": c.text, "correct_vs_gold": g["correct"]}) + "\n")
        print(f"{len(qs)} questions x {a.k}; prompted-sample accuracy {n_ok / (len(qs) * a.k):.3f} (logged only, not used to filter)")
        return

    import torch
    from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(a.seed)
    byid = {q["id"]: q for q in qs}; pf = prefix_fn(a.task)
    exs = []
    for l in open(os.path.join(out, "samples.jsonl")):
        s = json.loads(l); q = byid[s["id"]]
        t_prompt = tok.encode(render(tok, a.student, pf(q) + G.format_prompt(q)), add_special_tokens=False)
        s_prompt = tok.encode(render(tok, a.student, G.format_prompt(q)), add_special_tokens=False)
        reply = tok.encode(s["reply"], add_special_tokens=False) + [tok.convert_tokens_to_ids("<end_of_turn>") if "gemma" in a.student.lower() else tok.convert_tokens_to_ids("<|im_end|>")]
        exs.append((t_prompt, s_prompt, reply))
    model = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.float32, attn_implementation="eager" if "gemma" in a.student.lower() else "sdpa").cuda()
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    accum = 8; steps = math.ceil(len(exs) * a.epochs / accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * steps)), steps)
    order = []
    for e in range(math.ceil(a.epochs)):
        p = list(range(len(exs))); random.Random(a.seed * 100 + e).shuffle(p); order += p
    order = order[:steps * accum]
    log = open(os.path.join(out, "train_log.jsonl"), "w"); t0 = time.time(); tokens = 0; run = 0.0
    model.train()
    for i, idx in enumerate(order):
        tp, sp, rep = exs[idx]; R = len(rep)
        with torch.no_grad(), model.disable_adapter(), torch.autocast("cuda", dtype=torch.bfloat16):
            tl = model(input_ids=torch.tensor([tp + rep]).cuda()).logits[0, len(tp) - 1:len(tp) - 1 + R].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sl = model(input_ids=torch.tensor([sp + rep]).cuda()).logits[0, len(sp) - 1:len(sp) - 1 + R].float()
        lt = torch.log_softmax(tl, -1); ls = torch.log_softmax(sl, -1)
        kl = (lt.exp() * (lt - ls)).sum(-1).mean()
        (kl / accum).backward(); run += kl.item(); tokens += len(tp) + len(sp) + 2 * R
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad()
            step = (i + 1) // accum
            if step % 10 == 0 or step == steps:
                rec = {"step": step, "of": steps, "kl": round(run / accum, 4), "sec": round(time.time() - t0)}
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
            run = 0.0
    model.save_pretrained(os.path.join(out, "adapter"))
    json.dump({"baseline": "B5", "task": a.task, "student": a.student, "n_questions": len(qs), "samples_per_q": a.k, "n_examples": len(exs),
               "epochs": a.epochs, "lr": a.lr, "r": a.r, "steps": steps, "forward_tokens": tokens, "train_seconds": round(time.time() - t0),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "seed": a.seed}, open(os.path.join(out, "config.json"), "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
