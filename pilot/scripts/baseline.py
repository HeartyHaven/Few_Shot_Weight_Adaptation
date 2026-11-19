"""WP0 pre-teaching baselines (B0 = question only, B1 = procedure in the prompt).

Usage: python baseline.py <model_repo> <out_dir>
Greedy decoding, fixed prompts. Writes per-item outputs and a summary json.
"""
import json, os, re, sys, time
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO, OUT = sys.argv[1], sys.argv[2]
LIMIT = int(os.environ.get("LIMIT", "0"))
FREE_TOKENS = int(os.environ.get("FREE_TOKENS", "512"))
MODES = os.environ.get("MODES", "direct,free").split(",")
SUF = os.environ.get("SUFFIX", "")
HF = os.environ["HF_HOME"]
os.makedirs(OUT, exist_ok=True)
tag = REPO.split("/")[-1]

tok = AutoTokenizer.from_pretrained(REPO, cache_dir=HF)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(REPO, cache_dir=HF, torch_dtype=torch.bfloat16).cuda().eval()


def render(user):
    kw = {"enable_thinking": False} if "Qwen3" in REPO else {}
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, **kw)


@torch.no_grad()
def generate(prompts, max_new, bs):
    texts = [render(p) for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = [None] * len(texts)
    n_in = n_out = 0
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        new = gen[:, enc.input_ids.shape[1]:]
        n_in += int(enc.attention_mask.sum()); n_out += int((new != tok.pad_token_id).sum())
        for i, o in zip(idx, tok.batch_decode(new, skip_special_tokens=True)):
            out[i] = o
    return out, n_in, n_out


NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_num(text, free):
    if free and "final answer" in text.lower():
        text = text[text.lower().rindex("final answer"):]
        m = NUM.search(text.replace(",", ""))
        return float(m.group()) if m else None
    ms = NUM.findall(text.replace(",", ""))
    if not ms:
        return None
    return float(ms[-1] if free else ms[0])


OPT = re.compile(r"\(([A-G])\)")


def parse_opt(text, free):
    if free and "final answer" in text.lower():
        text = text[text.lower().rindex("final answer"):]
        m = OPT.search(text) or re.search(r"\b([A-G])\b", text[len("final answer"):])
        return f"({m.group(1)})" if m else None
    ms = OPT.findall(text)
    if not ms:
        m = re.match(r"\s*([A-G])\b", text)
        return f"({m.group(1)})" if m else None
    return f"({ms[-1] if free else ms[0]})"


def ci(correct):
    a = np.array(correct, dtype=float)
    rng = np.random.default_rng(0)
    bs = rng.choice(a, size=(1000, len(a)), replace=True).mean(1)
    return [round(float(a.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


# ---------------- T1: MedCalc rule-based calculators ----------------
mc = load_dataset("nsk7153/MedCalc-Bench-Verified")
test = [r for r in mc["test"] if r["Category"] in ("risk", "severity", "diagnosis")]
rules = {}
def extract_rule(e):
    """Keep the generic criteria list at the top of a worked explanation; drop the patient-specific part."""
    keep, seen = [], False
    for line in e.strip().split("\n"):
        numbered = bool(re.match(r"\s*\d+\.\s", line))
        sub = bool(re.match(r"\s+\S|\s*[-*•]", line))
        if seen and line.strip() and not numbered and not sub:
            if re.search(r"sum|calculat|total", line, re.I):
                keep.append(line)
            break
        seen = seen or numbered
        keep.append(line)
    return "\n".join(keep).strip()


for r in mc["one_shot"]:
    rules[r["Calculator ID"]] = extract_rule(r["Ground Truth Explanation"])
json.dump({str(k): v for k, v in rules.items()}, open(f"{OUT}/t1_rules_extracted.json", "w"), indent=1, ensure_ascii=False)

T1_MODE = {"direct": "Answer with only the final number, nothing else.",
           "free": "Work step by step, then finish with a line 'Final answer: <number>'."}


def t1_prompt(r, cond, mode):
    pre = ""
    if cond == "B1":
        pre = ("Scoring rule for this calculator:\n" + rules[r["Calculator ID"]] +
               "\nIf an item is not mentioned in the patient note, assume it is absent.\n\n")
    return f"{pre}Patient note:\n{r['Patient Note']}\n\nQuestion: {r['Question']}\n\n{T1_MODE[mode]}"


# ---------------- T2: BBH tracking shuffled objects ----------------
t2 = []
for k in ("three", "five", "seven"):
    for i, r in enumerate(load_dataset("lukaemon/bbh", f"tracking_shuffled_objects_{k}_objects")["test"]):
        t2.append({"sub": k, "id": f"{k}-{i}", **r})

T2_PROC = ("Procedure to follow:\n1. Write down the starting assignment as a table (person -> item).\n"
           "2. Process the swaps strictly in the order given, one at a time. For a swap between X and Y, exchange the "
           "two entries of X and Y and leave everyone else unchanged. After each swap, rewrite the whole table.\n"
           "3. Only after the last swap, read off the entry for the person asked about.\n"
           "4. Pick the option that matches that entry.\n\n")
T2_MODE = {"direct": "Answer with only the option letter in parentheses, for example (A).",
           "free": "Work step by step, then finish with a line 'Final answer: (X)'."}


def t2_prompt(r, cond, mode):
    return (T2_PROC if cond == "B1" else "") + r["input"] + "\n\n" + T2_MODE[mode]


if LIMIT:
    test, t2 = test[:LIMIT], t2[::max(1, len(t2) // LIMIT)][:LIMIT]

summary = {"model": REPO, "n_t1": len(test), "n_t2": len(t2), "results": {}, "per_calculator": {}, "cost": {}}
@torch.no_grad()
def forced_choice(prompts, n_opts, bs=32):
    """Prefill the reply with "(" and compare next-token logits over the option letters."""
    texts = [render(p) + "(" for p in prompts]
    letters = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCDEFG"]
    preds = []
    for s0 in range(0, len(texts), bs):
        enc = tok(texts[s0:s0 + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        lg = model(**enc).logits[:, -1].float()
        for j in range(lg.shape[0]):
            k = n_opts[s0 + j]
            preds.append("(" + "ABCDEFG"[int(lg[j, letters[:k]].argmax())] + ")")
    return preds


if os.environ.get("RULES_ONLY"):
    sys.exit()
for cond in ("B0", "B1"):
    for mode, max_new in (("direct", 16), ("free", FREE_TOKENS)):
        if mode not in MODES:
            continue
        key = f"{cond}_{mode}{SUF}"
        t0 = time.time()
        outs, n_in, n_out = generate([t1_prompt(r, cond, mode) for r in test], max_new, 8)
        rows, corr = [], []
        for r, o in zip(test, outs):
            v = parse_num(o, mode == "free")
            ok = v is not None and float(r["Lower Limit"]) <= v <= float(r["Upper Limit"])
            corr.append(ok)
            rows.append({"row": r["Row Number"], "calc_id": r["Calculator ID"], "calc": r["Calculator Name"],
                         "gold": r["Ground Truth Answer"], "pred": v, "correct": ok, "output": o})
        with open(f"{OUT}/t1_{key}_{tag}.jsonl", "w") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        summary["results"][f"T1_{key}"] = ci(corr)
        pc = {}
        for x in rows:
            pc.setdefault(x["calc"], []).append(x["correct"])
        summary["per_calculator"][key] = {k: round(sum(v) / len(v), 3) for k, v in pc.items()}
        summary["cost"][f"T1_{key}"] = {"in_tokens": n_in, "out_tokens": n_out, "sec": round(time.time() - t0, 1)}

        if mode == "direct":
            nopt = [{"three": 3, "five": 5, "seven": 7}[r["sub"]] for r in t2]
            fp = forced_choice([t2_prompt(r, cond, "direct") for r in t2], nopt)
            for k in ("three", "five", "seven", "all"):
                c = [p == r["target"] for p, r in zip(fp, t2) if k in (r["sub"], "all")]
                summary["results"][f"T2_{k}_{cond}_forced{SUF}"] = ci(c)
        t0 = time.time()
        outs, n_in, n_out = generate([t2_prompt(r, cond, mode) for r in t2], max_new, 32)
        rows = [{"id": r["id"], "sub": r["sub"], "gold": r["target"], "pred": parse_opt(o, mode == "free"), "output": o}
                for r, o in zip(t2, outs)]
        for x in rows:
            x["correct"] = x["pred"] == x["gold"]
        with open(f"{OUT}/t2_{key}_{tag}.jsonl", "w") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        for k in ("three", "five", "seven"):
            c = [x["correct"] for x in rows if x["sub"] == k]
            if c:
                summary["results"][f"T2_{k}_{key}"] = ci(c)
        summary["results"][f"T2_all_{key}"] = ci([x["correct"] for x in rows])
        summary["cost"][f"T2_{key}"] = {"in_tokens": n_in, "out_tokens": n_out, "sec": round(time.time() - t0, 1)}
        print(key, {k: v for k, v in summary["results"].items() if k.endswith(key)}, flush=True)
        json.dump(summary, open(f"{OUT}/baseline_{tag}{SUF}.json", "w"), indent=1, ensure_ascii=False)
print("DONE")
