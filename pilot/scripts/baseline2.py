"""Baseline pilot for three candidate harder tasks (unedited students, plain prompt, free answers).
  ddx  : DDXPlus differential diagnosis, 5 options (true disease + the 4 most likely other diseases of its differential)
  code : MBPP sanitized (test split), graded by running the unit tests
  math : MATH number theory, levels 1-3, graded on the final boxed answer
Usage: python pilot/scripts/baseline2.py <hf_repo> [n_per_task]
"""
import ast, json, os, random, re, subprocess, sys, tempfile
import pandas as pd, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
HF = "/mnt/data2/users/lang.gao/huggingface_cache"; os.environ["HF_HOME"] = HF
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
repo = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 150; tag = repo.split("/")[-1]
BS = int(os.environ.get("PILOT_BS", "16"))
dl = lambda r, f: hf_hub_download(r, f, repo_type="dataset", cache_dir=HF)

def build_ddx():
    ev = json.load(open(dl("aai530-group6/ddxplus", "release_evidences.json")))
    df = pd.read_csv(dl("aai530-group6/ddxplus", "test.csv")).sample(n=4000, random_state=11)
    rng = random.Random(11); items = []
    for _, r in df.iterrows():
        diff = [d for d, p in ast.literal_eval(r["DIFFERENTIAL_DIAGNOSIS"]) if d != r["PATHOLOGY"]][:4]
        if len(diff) < 4: continue
        lines = {}
        for code in ast.literal_eval(r["EVIDENCES"]):
            k, _, v = code.partition("_@_"); e = ev[k]; q = e["question_en"]
            if not v: lines.setdefault(q, []).append("yes")
            else:
                vm = e.get("value_meaning", {}).get(v, {}).get("en", v); lines.setdefault(q, []).append(str(vm))
        opts = diff + [r["PATHOLOGY"]]; rng.shuffle(opts)
        text = (f"Patient: {r['AGE']} years old, sex {r['SEX']}.\nFindings from the interview:\n" + "\n".join(f"- {q} {', '.join(a)}" for q, a in lines.items()) +
                "\n\nWhich diagnosis is most likely?\nOptions:\n" + "\n".join(f"({'ABCDE'[i]}) {o}" for i, o in enumerate(opts)))
        items.append({"id": len(items), "q": text, "answer": "(" + "ABCDE"[opts.index(r["PATHOLOGY"])] + ")", "group": r["PATHOLOGY"]})
        if len(items) >= N: break
    return items

def build_code():
    df = pd.read_parquet(dl("google-research-datasets/mbpp", "sanitized/test-00000-of-00001.parquet")).sample(n=N, random_state=11)
    return [{"id": int(r["task_id"]), "q": f"{r['prompt']}\nYour code must pass this test:\n{r['test_list'][0]}", "tests": list(r["test_list"]), "imports": list(r["test_imports"])} for _, r in df.iterrows()]

def build_math():
    df = pd.read_parquet(dl("EleutherAI/hendrycks_math", "number_theory/test-00000-of-00001.parquet"))
    df = df[df["level"].isin(["Level 1", "Level 2", "Level 3"])].sample(n=N, random_state=11)
    def boxed(s):
        i = s.rfind("\\boxed{"); d = 0; out = ""
        for ch in s[i + 7:]:
            if ch == "{": d += 1
            if ch == "}":
                if d == 0: break
                d -= 1
            out += ch
        return out
    return [{"id": i, "q": r["problem"], "answer": boxed(r["solution"]), "group": r["level"]} for i, (_, r) in enumerate(df.iterrows())]

INST = {"ddx": "\n\nThink step by step, then finish with a line 'Final answer: (X)'.",
        "code": "\n\nWrite the Python function. Put the complete code in one ```python block.",
        "math": "\n\nWork step by step, then finish with a line 'Final answer: <answer>'."}
norm = lambda s: re.sub(r"\\text\{([^}]*)\}|\\(?:left|right|,|!| )|\$|\s|dfrac", lambda m: m.group(1) or ("frac" if m.group(0) == "dfrac" else ""), str(s)).rstrip(".")

def grade(task, it, out):
    if task == "ddx":
        m = re.findall(r"Final answer:\s*\(?([A-E])\)?", out); return bool(m) and "(" + m[-1] + ")" == it["answer"]
    if task == "math":       # accept a "Final answer:" line or the last \\boxed{} near the end, tolerant of markdown/LaTeX wrappers
        def bx(t):
            k = t.rfind("\\boxed{")
            if k < 0: return None
            d = 0; o = ""
            for ch in t[k + 7:]:
                if ch == "{": d += 1
                if ch == "}":
                    if d == 0: break
                    d -= 1
                o += ch
            return o
        def nm(t):
            t = re.sub(r"\\text\{([^}]*)\}", r"\1", str(t)).replace("dfrac", "frac").replace("<answer>", "")
            return re.sub(r"\\left|\\right|\\,|\\!|\\ |\$|\*|\s|\\\(|\\\)|`", "", t).rstrip(".").lower()
        tail = out[-400:]; cands = []
        m = re.findall(r"[Ff]inal [Aa]nswer:?\**\s*(.+)", tail)
        if m: cands.append(bx(m[-1]) or m[-1].strip())
        if bx(tail) is not None: cands.append(bx(tail))
        return any(nm(c) == nm(it["answer"]) for c in cands)
    m = re.findall(r"```(?:python)?\n(.*?)(?:```|$)", out, re.S); m = [c for c in m if "def " in c] or m
    code = m[0] if m else out          # first block that defines a function; drop the model's own top-level tests/prints
    keep = []
    for line in code.split("\n"):
        if re.match(r"(assert |print\(|# ?(Test|Example))", line): break
        keep.append(line)
    code = "\n".join(keep)
    src = "\n".join(it["imports"]) + "\n" + code + "\n" + "\n".join(it["tests"]) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(src)
    try:
        r = subprocess.run(["unshare", "-rn", sys.executable, "-I", f.name], capture_output=True, timeout=15, cwd=tempfile.gettempdir()); return r.returncode == 0
    except subprocess.TimeoutExpired: return False
    finally: os.unlink(f.name)

tok = AutoTokenizer.from_pretrained(repo, cache_dir=HF); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(repo, cache_dir=HF, dtype=torch.bfloat16).cuda().eval()
if os.environ.get("TDWA_MEM_GB"):
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["TDWA_MEM_GB"]) * 2**30 / torch.cuda.get_device_properties(0).total_memory))
kw = {"enable_thinking": False} if "qwen" in repo.lower() else {}
@torch.no_grad()
def gen(users, max_new, bs=BS):
    texts = [tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, **kw) for u in users]
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i])); out = [None] * len(texts)
    s = 0
    while s < len(order):
        idx = order[s:s + bs]
        try:
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1: raise
            bs = max(1, bs // 2); continue
        for i, o in zip(idx, tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)): out[i] = o
        s += len(idx)
    return out
res = {}
for task, build in (("ddx", build_ddx), ("code", build_code), ("math", build_math)):
    items = build(); outs = gen([it["q"] + INST[task] for it in items], 1024)
    ok = [grade(task, it, o) for it, o in zip(items, outs)]
    res[task] = {"n": len(items), "acc": round(sum(ok) / len(ok), 4), "mean_prompt_chars": round(sum(len(it["q"]) for it in items) / len(items)), "mean_out_chars": round(sum(len(o) for o in outs) / len(outs))}
    with open(f"{ROOT}/pilot/out2/{task}__{tag}.jsonl", "w") as f:
        for it, o, k in zip(items, outs, ok): f.write(json.dumps({**{x: it[x] for x in it if x not in ("tests", "imports")}, "output": o, "correct": k}, ensure_ascii=False) + "\n")
    print(task, res[task], flush=True)
json.dump(res, open(f"{ROOT}/pilot/out2/summary__{tag}.json", "w"), indent=1)
