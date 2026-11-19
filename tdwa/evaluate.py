"""Independent evaluation of a finished teaching run on the hidden test sets.

Usage: python -m tdwa.evaluate --run runs/<name> [--gpu 0]
       python -m tdwa.evaluate --base --task t2 --student google/gemma-3-1b-it     (unedited student, cached)
Writes eval/<name>.json (+ per-item outputs under _secret/eval/<name>/).
Prompts are the plain official format (no procedure). Two answer modes: free (2048 new tokens) and, for
multiple choice, forced choice (reply prefilled with "(" and option-letter logits compared).
"""
import argparse, glob, json, math, os, sys, time

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tdwa import grade as G  # noqa: E402

HF = os.environ.get("TDWA_HF", "/mnt/data2/users/lang.gao/huggingface_cache")   # per-host override
os.environ.setdefault("HF_HOME", HF)


def ci(c):
    a = np.array(c, dtype=float); rng = np.random.default_rng(0)
    bs = rng.choice(a, size=(1000, len(a)), replace=True).mean(1)
    return {"acc": round(float(a.mean()), 4), "lo": round(float(np.percentile(bs, 2.5)), 4), "hi": round(float(np.percentile(bs, 97.5)), 4), "n": len(a)}


class Student:
    def __init__(self, repo, edit_dir=None):
        self.repo = repo
        self.tok = AutoTokenizer.from_pretrained(repo, cache_dir=HF); self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(repo, cache_dir=HF, dtype=getattr(torch, os.environ.get("TDWA_DTYPE", "float32"))).cuda().eval()   # TDWA_DTYPE=bfloat16 for large students
        self.n_changed = 0
        if edit_dir:
            summ = json.load(open(os.path.join(edit_dir, "summary.json")))
            new = load_file(os.path.join(edit_dir, "edit.safetensors")); new.pop("_empty", None)
            params = dict(self.model.named_parameters())
            for name, site in summ.get("extra_sites", {}).items():      # bias vectors the teacher added
                if name not in new or not bool(new[name].abs().sum()):   # an added bias that stayed zero has no effect
                    continue
                p = torch.nn.Parameter(new[name].cuda(), requires_grad=False); self.model.register_parameter(name, p)
                mod = self.model
                for part in site.split("."):
                    mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
                mod.register_forward_hook(lambda m, i, o, p=p: ((o[0] + p,) + tuple(o[1:])) if isinstance(o, tuple) else o + p)
                self.n_changed += p.numel()
            for name, t in new.items():
                if name in params:
                    self.n_changed += int((params[name].detach().cpu() != t).sum()); params[name].data.copy_(t.cuda())
        if os.environ.get("TDWA_MEM_GB"):      # shared GPUs: never take more than this, so teaching runs keep their room
            torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["TDWA_MEM_GB"]) * 2**30 / torch.cuda.get_device_properties(0).total_memory))
        self.fwd = 0
        self.model.get_input_embeddings().register_forward_hook(lambda m, i, o: setattr(self, "fwd", self.fwd + o.shape[0] * o.shape[1]))

    def render(self, user):
        kw = {"enable_thinking": False} if "qwen" in self.repo.lower() else {}
        return self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, **kw)

    @torch.no_grad()
    def generate(self, users, max_new, bs=int(os.environ.get("TDWA_BS", "16"))):
        texts = [self.render(u) for u in users]; order = sorted(range(len(texts)), key=lambda i: -len(texts[i])); out = [None] * len(texts)
        s = 0
        while s < len(order):
            idx = order[s:s + bs]
            try:
                enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                gen = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
            except torch.OutOfMemoryError:           # shared GPU: retry with a smaller batch (greedy decoding, same outputs)
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2); print("OOM, batch size ->", bs, flush=True); continue
            for i, o in zip(idx, self.tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
                out[i] = o
            s += len(idx)
        return out

    @torch.no_grad()
    def forced(self, users, n_opts, bs=16):
        texts = [self.render(u) + "(" for u in users]; preds = []
        L = [self.tok.encode(c, add_special_tokens=False)[0] for c in "ABCDEFGHI"]
        s = 0
        while s < len(texts):
            try:
                enc = self.tok(texts[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                lg = self.model(**enc).logits[:, -1].float().cpu()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2); continue
            preds += ["(" + "ABCDEFGHI"[int(lg[j, L[:n_opts[s + j]]].argmax())] + ")" for j in range(lg.shape[0])]
            s += lg.shape[0]
        return preds

    @torch.no_grad()
    def ppl(self, text):
        ids = self.tok(text, return_tensors="pt").input_ids.cuda()
        return math.exp(self.model(ids, labels=ids).loss.item())


def retention(st, out_rows):
    """Side-effect suite that needs no extra downloads: generic-text perplexity, ARC-Challenge and HellaSwag
    (forced choice, items disjoint from the teacher-visible canary), and two-digit/three-digit arithmetic."""
    import random
    from datasets import load_dataset
    res = {"ppl_generic": round(st.ppl(open(os.path.join(ROOT, "data", "generic.txt")).read()), 4)}
    rng = random.Random(9909)
    arc = [r for r in load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"] if r["choices"]["label"][0] == "A" and len(r["choices"]["label"]) <= 5]
    arc = rng.sample(arc, 300)
    users = [r["question"] + "\n" + "\n".join(f"({l}) {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"])) +
             "\nAnswer with the option letter in parentheses." for r in arc]
    p = st.forced(users, [len(r["choices"]["label"]) for r in arc])
    res["arc_challenge"] = ci([a == f"({r['answerKey']})" for a, r in zip(p, arc)])
    hs = rng.sample(list(load_dataset("Rowan/hellaswag")["validation"]), 300)
    users = [f"Choose the most plausible continuation.\n{r['ctx']}\n" + "\n".join(f"({'ABCD'[i]}) {e}" for i, e in enumerate(r["endings"])) +
             "\nAnswer with the option letter in parentheses." for r in hs]
    p = st.forced(users, [4] * len(hs))
    res["hellaswag"] = ci([a == f"({'ABCD'[int(r['label'])]})" for a, r in zip(p, hs)])
    items = []
    for _ in range(200):
        a, b = rng.randint(11, 999), rng.randint(11, 999); op = rng.choice("+-")
        items.append((f"What is {a} {op} {b}? Answer with the number only.", str(a + b if op == "+" else a - b)))
    outs = st.generate([u for u, _ in items], 12)
    res["arithmetic"] = ci([ans in o.replace(",", "") for (u, ans), o in zip(items, outs)])
    out_rows["retention_arith"] = [{"q": u, "gold": g, "output": o} for (u, g), o in zip(items, outs)]
    return res


def evaluate(st, task, out_dir):
    os.makedirs(out_dir, exist_ok=True); res = {}; rows_extra = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "_secret", task, "test*.jsonl"))):
        name = os.path.basename(path)[:-6]; items = [json.loads(l) for l in open(path)]
        t0 = time.time(); f0 = st.fwd
        outs = st.generate([G.format_prompt(it) for it in items], 2048)
        rows = [{"id": it["id"], "group": it["group"], "taught": it.get("taught"), "answer": it["answer"], "output": o, **G.grade(it, o)} for it, o in zip(items, outs)]
        if task in G.OPT_TASKS:
            fp = st.forced([G.format_prompt(it, "direct") for it in items], [it.get("n_options") or it["n_objects"] for it in items])
            for r, p, it in zip(rows, fp, items):
                r["forced_pred"] = p; r["forced_correct"] = p == it["answer"]
        with open(os.path.join(out_dir, name + ".jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        d = {"free": ci([r["correct"] for r in rows]), "no_final_answer_rate": round(sum(r["pred"] is None for r in rows) / len(rows), 4),
             "mean_output_chars": round(sum(len(r["output"]) for r in rows) / len(rows)), "sec": round(time.time() - t0), "forward_tokens": st.fwd - f0, "by_group": {}}
        if task in G.OPT_TASKS:
            d["forced"] = ci([r["forced_correct"] for r in rows])
        keyf = (lambda r: ("taught/" if r["taught"] else "untaught/") + r["group"]) if task == "t1" else (lambda r: r["group"])
        for g in sorted({keyf(r) for r in rows}):
            sub = [r for r in rows if keyf(r) == g]
            d["by_group"][g] = {"free": ci([r["correct"] for r in sub])}
            if task in G.OPT_TASKS:
                d["by_group"][g]["forced"] = ci([r["forced_correct"] for r in sub])
        if task == "t1":
            for k, flag in (("taught", True), ("untaught", False)):
                d[k] = ci([r["correct"] for r in rows if r["taught"] == flag])
        res[name] = d; print(name, d["free"], d.get("forced"), flush=True)
    res["retention"] = retention(st, rows_extra); print("retention", res["retention"], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--base", action="store_true"); ap.add_argument("--task"); ap.add_argument("--student")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "eval"), exist_ok=True)
    if a.base:
        name = f"base__{a.task}__{a.student.split('/')[-1]}" + ("__bf16" if os.environ.get("TDWA_DTYPE") == "bfloat16" else ""); st = Student(a.student); task = a.task; meta = {"student": a.student, "task": task, "edited": False}
    else:
        run = a.run.rstrip("/"); cfg = json.load(open(os.path.join(run, "config.json"))); name = os.path.basename(run); task = cfg["task"]
        st = Student(cfg["student"], os.path.join(run, "final"))
        meta = {"student": cfg["student"], "task": task, "edited": True, "teacher": cfg["teacher"], "values_changed": st.n_changed,
                "run_summary": json.load(open(os.path.join(run, "final", "summary.json"))), "teacher_usage": cfg.get("teacher_usage"),
                "turns": cfg.get("turns"), "wall_seconds": cfg.get("wall_seconds"), "validation_looks": cfg.get("validation_looks")}
    res = evaluate(st, task, os.path.join(ROOT, "_secret", "eval", name))
    json.dump({**meta, "results": res}, open(os.path.join(ROOT, "eval", name + ".json"), "w"), indent=1, default=str)
    print("wrote", os.path.join("eval", name + ".json"))


if __name__ == "__main__":
    main()
