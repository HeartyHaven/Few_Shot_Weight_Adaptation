"""Broader side-effect check (RET-FULL, first part): MMLU and GSM8K with vLLM, for the base students and the LoRA controls.

  MMLU  : cais/mmlu "all" test, 20 items per subject (57 subjects, 1140 items, seed 1212), zero-shot, forced choice over A-D
          (one greedy step restricted to the letter tokens after "(" -- same mechanism as evaluate_vllm.forced).
  GSM8K : openai/gsm8k main test, 500 items (seed 1212), zero-shot, "Work step by step, then finish with a line
          'Final answer: <number>'.", 512 new tokens, graded on the last number after "Final answer" (else last number).
IFEval is not included yet (needs the official instruction checkers).
Usage: python -m tdwa.retention_vllm --student Qwen/Qwen3-4B [--lora baselines/<run>/adapter] --name <name>
Writes eval_ret/<name>.json (aggregates + bootstrap CI) and per-item outputs under _secret/eval_ret/<name>/.
"""
import argparse, json, os, random, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tdwa.evaluate import ci  # noqa: E402
from tdwa.evaluate_vllm import VStudent  # noqa: E402


def num(s):
    """First number after the last 'Final answer', else the last number in the reply."""
    s = s.replace(",", "")
    if "final answer" in s.lower():
        m = re.findall(r"-?\d+(?:\.\d+)?", s[s.lower().rindex("final answer"):])
        if m:
            return float(m[0])
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True); ap.add_argument("--lora"); ap.add_argument("--name", required=True); ap.add_argument("--mem", type=float, default=0.4)
    a = ap.parse_args()
    from datasets import load_dataset
    st = VStudent(a.student, None, 1, mem=a.mem, lora=a.lora)
    rng = random.Random(1212); res = {}; out_dir = os.path.join(ROOT, "_secret", "eval_ret", a.name); os.makedirs(out_dir, exist_ok=True)
    mm = load_dataset("cais/mmlu", "all")["test"]; by = {}
    for r in mm:
        by.setdefault(r["subject"], []).append(r)
    items = [x for s in sorted(by) for x in rng.sample(by[s], min(20, len(by[s])))]
    users = [f"{r['question']}\n" + "\n".join(f"({'ABCD'[i]}) {c}" for i, c in enumerate(r["choices"])) + "\nAnswer with the option letter in parentheses." for r in items]
    p = st.forced(users, [4] * len(items)); ok = [x == f"({'ABCD'[r['answer']]})" for x, r in zip(p, items)]
    res["mmlu"] = ci(ok)
    json.dump([{"subject": r["subject"], "pred": x, "gold": "ABCD"[r["answer"]]} for r, x in zip(items, p)], open(os.path.join(out_dir, "mmlu.json"), "w"))
    gs = load_dataset("openai/gsm8k", "main")["test"]; g = rng.sample(list(gs), 500)
    outs = st.generate([r["question"] + "\n\nWork step by step, then finish with a line 'Final answer: <number>'." for r in g], 512)
    gold = [float(r["answer"].split("####")[-1].strip().replace(",", "")) for r in g]
    ok = [(lambda v: v is not None and abs(v - y) < 1e-6)(num(o)) for o, y in zip(outs, gold)]
    res["gsm8k"] = ci(ok)
    json.dump([{"q": r["question"], "gold": y, "out": o, "ok": k} for r, y, o, k in zip(g, gold, outs, ok)], open(os.path.join(out_dir, "gsm8k.json"), "w"))
    os.makedirs(os.path.join(ROOT, "eval_ret"), exist_ok=True)
    json.dump({"student": a.student, "lora": a.lora, "engine": "vllm", "results": res}, open(os.path.join(ROOT, "eval_ret", a.name + ".json"), "w"), indent=1)
    print(a.name, res)


if __name__ == "__main__":
    main()
