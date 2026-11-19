"""vLLM version of tdwa/evaluate.py (same prompts, graders, hidden test sets and retention suite; much faster decoding).

Usage:
  python -m tdwa.evaluate_vllm --task t1 --student google/gemma-3-1b-it [--model-path <merged dir>] [--name NAME] [--tp 2]
  python -m tdwa.evaluate_vllm --new-tasks --student google/gemma-3-27b-it --tp 2      (DDXPlus / MBPP / MATH pilot, as pilot/scripts/baseline2.py)

Differences from tdwa/evaluate.py, which must be stated when numbers are compared:
  * weights run in bfloat16 (vLLM kernels); the main teaching evaluations used float32 HF generate. Greedy outputs can
    diverge after the first near-tie, so compare edited/LoRA students only with a base student measured by THIS script.
  * forced choice = one greedy step restricted to the option-letter tokens (identical argmax to the HF logits comparison).
  * perplexity from vLLM prompt logprobs over the same token ids.
Writes eval/<name>.json (engine field = "vllm") and per-item outputs under _secret/eval/<name>/.
"""
import argparse, json, math, os, random, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tdwa import grade as G  # noqa: E402
from tdwa.evaluate import ci  # noqa: E402   (bootstrap CI, same seed)

import glob  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402


class VStudent:
    def __init__(self, repo, path=None, tp=1, dtype="bfloat16", mem=0.85, max_len=8192):
        self.repo = repo; src = path or repo
        self.tok = AutoTokenizer.from_pretrained(src)
        self.llm = LLM(model=src, tokenizer=src, tensor_parallel_size=tp, dtype=dtype, gpu_memory_utilization=mem,
                       max_model_len=max_len, seed=0, enable_prefix_caching=True)
        self.fwd = 0

    def render(self, user):
        kw = {"enable_thinking": False} if "qwen" in self.repo.lower() else {}
        return self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, **kw)

    def _ids(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def generate(self, users, max_new):
        ps = [TokensPrompt(prompt_token_ids=self._ids(self.render(u))) for u in users]
        outs = self.llm.generate(ps, SamplingParams(temperature=0, max_tokens=max_new), use_tqdm=False)
        self.fwd += sum(len(p["prompt_token_ids"]) + len(o.outputs[0].token_ids) for p, o in zip(ps, outs))
        return [o.outputs[0].text for o in outs]

    def forced(self, users, n_opts):
        L = [self.tok.encode(c, add_special_tokens=False)[0] for c in "ABCDEFGHI"]
        ps = [TokensPrompt(prompt_token_ids=self._ids(self.render(u) + "(")) for u in users]
        sp = [SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=L[:n]) for n in n_opts]
        outs = self.llm.generate(ps, sp, use_tqdm=False)
        self.fwd += sum(len(p["prompt_token_ids"]) for p in ps)
        return ["(" + "ABCDEFGHI"[L.index(o.outputs[0].token_ids[0])] + ")" for o in outs]

    def ppl(self, text):
        ids = self.tok(text).input_ids             # same ids as HF tok(text) in evaluate.py (special tokens added)
        o = self.llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)[0]
        lp = [d[t].logprob for t, d in zip(ids[1:], o.prompt_logprobs[1:])]
        return math.exp(-sum(lp) / len(lp))


def retention(st, out_rows):
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
    os.makedirs(out_dir, exist_ok=True); res = {}; extra = {}
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
    res["retention"] = retention(st, extra); print("retention", res["retention"], flush=True)
    return res


def new_tasks(st, tag, n=150):
    """DDXPlus / MBPP / MATH-NT pilot with exactly the item builders and graders of pilot/scripts/baseline2.py."""
    src = open(os.path.join(ROOT, "pilot", "scripts", "baseline2.py")).read()
    src = src[:src.index("tok = AutoTokenizer")]            # builders + graders only (no HF model load)
    g = {"__file__": os.path.join(ROOT, "pilot", "scripts", "baseline2.py")}
    argv = sys.argv; sys.argv = ["baseline2.py", st.repo, str(n)]
    try:
        exec(compile(src, "baseline2.py", "exec"), g)
    finally:
        sys.argv = argv
    res = {}; os.makedirs(os.path.join(ROOT, "pilot", "out2"), exist_ok=True)
    for task, build in (("ddx", g["build_ddx"]), ("code", g["build_code"]), ("math", g["build_math"])):
        items = build(); outs = st.generate([it["q"] + g["INST"][task] for it in items], 1024)
        ok = [g["grade"](task, it, o) for it, o in zip(items, outs)]
        res[task] = {"n": len(items), "acc": round(sum(ok) / len(ok), 4), "mean_prompt_chars": round(sum(len(it["q"]) for it in items) / len(items)),
                     "mean_out_chars": round(sum(len(o) for o in outs) / len(outs)), "engine": "vllm-bf16"}
        with open(f"{ROOT}/pilot/out2/{task}__{tag}.jsonl", "w") as f:
            for it, o, k in zip(items, outs, ok):
                f.write(json.dumps({**{x: it[x] for x in it if x not in ("tests", "imports")}, "output": o, "correct": k}, ensure_ascii=False) + "\n")
        print(task, res[task], flush=True)
    json.dump(res, open(f"{ROOT}/pilot/out2/summary__{tag}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True); ap.add_argument("--model-path"); ap.add_argument("--task"); ap.add_argument("--name")
    ap.add_argument("--tp", type=int, default=1); ap.add_argument("--mem", type=float, default=0.85); ap.add_argument("--new-tasks", action="store_true")
    ap.add_argument("--meta", default="{}", help="extra JSON merged into the output metadata")
    a = ap.parse_args()
    st = VStudent(a.student, a.model_path, a.tp, mem=a.mem)
    tag = a.student.split("/")[-1]
    if a.new_tasks:
        new_tasks(st, tag); return
    name = a.name or f"base__{a.task}__{tag}__vllm"
    os.makedirs(os.path.join(ROOT, "eval"), exist_ok=True)
    t0 = time.time()
    res = evaluate(st, a.task, os.path.join(ROOT, "_secret", "eval", name))
    meta = {"student": a.student, "task": a.task, "edited": a.model_path is not None, "engine": "vllm", "dtype": "bfloat16",
            "model_path": a.model_path, "wall_seconds": round(time.time() - t0), **json.loads(a.meta)}
    json.dump({**meta, "results": res}, open(os.path.join(ROOT, "eval", name + ".json"), "w"), indent=1, default=str)
    print("wrote", os.path.join("eval", name + ".json"))


if __name__ == "__main__":
    main()
