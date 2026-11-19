"""Gradient baselines B2 (few-shot LoRA on the 100 teacher-visible dev items) and B4 (LoRA on all training data).

Usage: python -m tdwa.baseline_lora --task t1 --student google/gemma-3-1b-it --data few|full [--epochs N]
Writes baselines/<name>/{config.json, train_log.jsonl, adapter/, merged/}; evaluate afterwards with
  python -m tdwa.evaluate_vllm --task t1 --student <repo> --model-path baselines/<name>/merged --name <name>

Training targets (assistant turn, loss on the reply only; prompt = the exact evaluation prompt, free mode):
  t1: MedCalc 'Ground Truth Explanation' + "\nFinal answer: <answer>"
  t2: the ground-truth swap trace written out step by step + "Final answer: (X)"
Data:
  few : data/<task>/dev.jsonl (100 items) -- the same items the teacher sees
  full: t1 = every MedCalc-Bench-Verified train row (all calculators, ~10k, DESIGN 9.4 B4) minus the 120 hidden-validation rows;
        t2 = 5000 freshly generated items (seed 9191, 3/5/7 objects in the dev proportions 20/40/40), disjoint generator seed.
Hyper-parameters are fixed in advance (no tuning on validation or test): r=16, alpha=32, dropout 0.05, all attention+MLP
projections, AdamW lr 2e-4, cosine with 3% warm-up, effective batch 16 (few: batch 8), few = 10 epochs, full = 1 epoch.
"""
import argparse, json, math, os, random, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tdwa import grade as G  # noqa: E402
from tdwa import data as D  # noqa: E402


def t2_target(it):
    lines = ["Initial: " + ", ".join(f"{p}: {o}" for p, o in it["trace"][0].items()) + "."]
    for k, ((a, b), st) in enumerate(zip(it["swaps"], it["trace"][1:]), 1):
        lines.append(f"Swap {k} ({a} and {b}): " + ", ".join(f"{p}: {o}" for p, o in st.items()) + ".")
    lines.append(f"At the end, {it['asked']} has {it['trace'][-1][it['asked']]}.")
    return "\n".join(lines) + f"\nFinal answer: {it['answer']}"


def t1_target(it):
    return it["explanation"].strip() + f"\nFinal answer: {it['answer']}"


def load_items(task, which):
    if which == "few":
        return [json.loads(l) for l in open(os.path.join(ROOT, "data", task, "dev.jsonl"))]
    if task == "t2":
        rng = random.Random(9191); out = []
        for n, c in ((3, 1000), (5, 2000), (7, 2000)):
            out += [D.make_item(rng, n, None, f"full-{n}-{i}") for i in range(c)]
        return out
    from datasets import load_dataset
    val_ids = {json.loads(l)["id"].split("-", 1)[1] for l in open(os.path.join(ROOT, "_secret", "t1", "val.jsonl"))}
    out = []
    for r in load_dataset("nsk7153/MedCalc-Bench-Verified")["train"]:
        if str(r["Row Number"]) in val_ids:
            continue
        out.append({"id": f"train-{r['Row Number']}", "task": "t1", "group": r["Calculator Name"], "answer": r["Ground Truth Answer"],
                    "question": f"Patient note:\n{r['Patient Note']}\n\nQuestion: {r['Question']}", "explanation": r["Ground Truth Explanation"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--student", required=True); ap.add_argument("--data", choices=["few", "full"], required=True)
    ap.add_argument("--epochs", type=float); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=3072); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    tag = a.student.split("/")[-1]; bname = {"few": "B2", "full": "B4"}[a.data]
    name = f"{bname}__{a.task}__{tag}__s{a.seed}"; out = os.path.join(ROOT, "baselines", name); os.makedirs(out, exist_ok=True)
    epochs = a.epochs or (10 if a.data == "few" else 1)
    bs, accum = ((8, 1) if a.task == "t2" else (2, 4)) if a.data == "few" else (2, 8)   # t1 prompts are long; Gemma vocab is 262k

    tok = AutoTokenizer.from_pretrained(a.student)
    kw = {"enable_thinking": False} if "qwen" in a.student.lower() else {}
    items = load_items(a.task, a.data); tgt = t1_target if a.task == "t1" else t2_target
    exs, skipped = [], 0
    for it in items:
        user = G.format_prompt(it)
        prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, **kw)
        full = tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": tgt(it)}], tokenize=False, **kw)
        if not full.startswith(prompt):
            full = prompt + tgt(it) + (tok.eos_token or "")
        p_ids = tok.encode(prompt, add_special_tokens=False); f_ids = tok.encode(full, add_special_tokens=False)
        if len(f_ids) > a.max_len:
            skipped += 1; continue
        exs.append((f_ids, len(p_ids)))
    print(f"{len(exs)} examples ({skipped} longer than {a.max_len} tokens skipped)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.float32, attn_implementation="eager" if "gemma" in a.student.lower() else "sdpa").cuda()
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg); model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(exs) * epochs / (bs * accum))
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * steps)), steps)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    log = open(os.path.join(out, "train_log.jsonl"), "w"); t0 = time.time(); step = 0; tokens = 0
    order = []
    for e in range(math.ceil(epochs)):
        perm = list(range(len(exs))); random.Random(a.seed * 100 + e).shuffle(perm); order += perm
    order = order[:steps * bs * accum]
    model.train()
    for s in range(0, len(order), bs):
        batch = [exs[i] for i in order[s:s + bs]]; L = max(len(f) for f, _ in batch)
        ids = torch.full((len(batch), L), pad); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, (f, p) in enumerate(batch):
            ids[j, :len(f)] = torch.tensor(f); att[j, :len(f)] = 1; lab[j, p:len(f)] = torch.tensor(f[p:])
        tokens += int(att.sum())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss / accum
        loss.backward()
        if (s // bs + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); step += 1
            if step % 5 == 0 or step == steps:
                rec = {"step": step, "of": steps, "loss": round(loss.item() * accum, 4), "lr": sch.get_last_lr()[0], "sec": round(time.time() - t0)}
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    model.save_pretrained(os.path.join(out, "adapter"))
    merged = model.merge_and_unload().to(torch.bfloat16)
    merged.save_pretrained(os.path.join(out, "merged")); tok.save_pretrained(os.path.join(out, "merged"))
    json.dump({"baseline": bname, "task": a.task, "student": a.student, "data": a.data, "n_examples": len(exs), "skipped_too_long": skipped,
               "epochs": epochs, "lr": a.lr, "r": a.r, "alpha": 2 * a.r, "batch": bs * accum, "steps": steps, "train_tokens": tokens,
               "backward_token_cost": 3 * tokens, "train_seconds": round(time.time() - t0), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "seed": a.seed}, open(os.path.join(out, "config.json"), "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
