"""Difficulty pilot for candidate hard vertical tasks (unedited students, plain zero-shot prompt, vLLM bf16, greedy).

  math5   : MATH (EleutherAI/hendrycks_math) Level 5 test, 50 each from number_theory, counting_and_probability,
            intermediate_algebra, precalculus; graded on the final answer with the tolerant extractor of baseline2.py
  crux    : CRUXEval-O (cruxeval-org/cruxeval), 200 of 800; predict f(input); graded by literal equality
  mceq    : MedCalc-Bench-Verified test, formula-based calculators (categories lab / physical / dosage), 200 items;
            graded within the dataset's lower/upper limits (the benchmark's +-5% tolerance)
  finqa   : FinQA test (ChanceFocus/flare-finqa copy), 200 of 1147; numeric answer, correct if within 1% relative
            (or 0.005 absolute), percent/fraction scale mismatch (x100) also accepted
Usage: python pilot/scripts/hardtask_pilot.py <hf_repo> [n_per_task]
Writes pilot/out2/hard__<tag>.json (+ per-item jsonl).
"""
import ast, json, os, random, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tdwa.evaluate_vllm import VStudent  # noqa: E402
from tdwa.evaluate import ci  # noqa: E402
from datasets import load_dataset  # noqa: E402

repo = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 200; tag = repo.split("/")[-1]
rng = random.Random(4242)


def boxed(t):
    k = t.rfind("\\boxed{")
    if k < 0:
        return None
    d = 0; o = ""
    for ch in t[k + 7:]:
        if ch == "{": d += 1
        if ch == "}":
            if d == 0: break
            d -= 1
        o += ch
    return o


def nm(t):
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", str(t)).replace("dfrac", "frac").replace("tfrac", "frac")
    return re.sub(r"\\left|\\right|\\,|\\!|\\ |\$|\*|\s|\\\(|\\\)|`|\^\{\\circ\}|\^\\circ", "", t).rstrip(".").lower()


def after_final(out):
    """Text after the LAST 'final answer' (any case), without markdown fences/backticks/bold; first non-empty line."""
    lo = out.lower()
    if "final answer" not in lo:
        return None
    t = out[lo.rindex("final answer") + len("final answer"):]
    t = re.sub(r"^[\s:*#]*", "", t)
    t = re.sub(r"```[a-zA-Z]*", "\n", t).replace("`", "").replace("**", "")
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    return lines[0] if lines else None


def g_math(it, out):
    tail = out[-800:]; cands = []
    a = after_final(out)
    if a: cands.append(boxed(a) or a.strip().strip("$"))
    if boxed(tail) is not None: cands.append(boxed(tail))
    return any(nm(c) == nm(it["answer"]) for c in cands)


def g_crux(it, out):
    s = after_final(out) or (out.strip().splitlines()[-1] if out.strip() else "")
    s = s.strip().strip("`").strip()
    s = re.sub(r"^assert\s+f\(.*\)\s*==\s*", "", s)
    try:
        return ast.literal_eval(s) == ast.literal_eval(it["answer"])
    except Exception:
        return s == it["answer"]


def num(out):
    s = out.replace(",", "")
    if "final answer" in s.lower():
        m = re.findall(r"-?\d+(?:\.\d+)?", s[s.lower().rindex("final answer"):])
        if m: return float(m[0])
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def g_mc(it, out):
    v = num(out); return v is not None and it["lo"] <= v <= it["hi"]


def g_fin(it, out):
    v = num(out); y = it["answer"]
    if v is None: return False
    return any(abs(c - y) <= max(0.005, 0.01 * abs(y)) for c in (v, v / 100, v * 100))


def build():
    T = {}
    xs = []
    for sub in ("number_theory", "counting_and_probability", "intermediate_algebra", "precalculus"):
        d = [r for r in load_dataset("EleutherAI/hendrycks_math", sub)["test"] if r["level"] == "Level 5"]
        xs += [{"q": r["problem"], "answer": boxed(r["solution"]), "group": sub} for r in rng.sample(d, N // 4)]
    T["math5"] = (xs, "\n\nWork step by step, then finish with a line 'Final answer: <answer>'.", g_math, 2048)
    cx = rng.sample(list(load_dataset("cruxeval-org/cruxeval")["test"]), N)
    T["crux"] = ([{"q": f"Here is a Python function:\n```python\n{r['code']}\n```\nWhat is the value of f({r['input']})?", "answer": r["output"], "group": "crux"} for r in cx],
                 "\n\nReason step by step, then finish with a line 'Final answer: <the exact Python value>'.", g_crux, 1024)
    mc = [r for r in load_dataset("nsk7153/MedCalc-Bench-Verified")["test"] if r["Category"] in ("lab", "physical", "dosage")]
    mc = rng.sample(mc, min(N, len(mc)))
    T["mceq"] = ([{"q": f"Patient note:\n{r['Patient Note']}\n\nQuestion: {r['Question']}", "lo": float(r["Lower Limit"]), "hi": float(r["Upper Limit"]),
                   "answer": r["Ground Truth Answer"], "group": r["Calculator Name"]} for r in mc],
                 "\n\nWork step by step, then finish with a line 'Final answer: <number>'.", g_mc, 1024)
    fq = [r for r in load_dataset("ChanceFocus/flare-finqa")["test"]]
    fq = rng.sample(fq, N); items = []
    for r in fq:
        try:
            y = float(str(r["answer"]).replace("%", "").replace(",", "").replace("$", ""))
        except ValueError:
            continue
        items.append({"q": r["query"], "answer": y, "group": "finqa"})
    T["finqa"] = (items, "\n\nWork step by step, then finish with a line 'Final answer: <number>'.", g_fin, 1024)
    return T


if len(sys.argv) > 2 and sys.argv[2] == "regrade":       # re-score saved outputs with the current graders
    G = {"math5": g_math, "crux": g_crux, "mceq": g_mc, "finqa": g_fin}; res = json.load(open(f"{ROOT}/pilot/out2/hard__{tag}.json"))
    for name, g in G.items():
        rows = [json.loads(l) for l in open(f"{ROOT}/pilot/out2/hard_{name}__{tag}.jsonl")]
        ok = [bool(g(r, r["output"])) for r in rows]
        groups = {}
        for r, k in zip(rows, ok):
            groups.setdefault(r["group"], []).append(k)
        res[name].update(ci(ok)); res[name]["by_group"] = {k: round(sum(v) / len(v), 3) for k, v in groups.items()} if name == "math5" else None
        print(name, res[name]["acc"])
    json.dump(res, open(f"{ROOT}/pilot/out2/hard__{tag}.json", "w"), indent=1); sys.exit()
T = build()
st = VStudent(repo, None, 1, mem=float(os.environ.get("PILOT_MEM", "0.8")), max_len=8192)
res = {}; os.makedirs(f"{ROOT}/pilot/out2", exist_ok=True)
for name, (items, inst, grade, mx) in T.items():
    outs = st.generate([it["q"] + inst for it in items], mx)
    ok = [bool(grade(it, o)) for it, o in zip(items, outs)]
    groups = {}
    for it, k in zip(items, ok):
        groups.setdefault(it["group"], []).append(k)
    res[name] = {**ci(ok), "by_group": {g: round(sum(v) / len(v), 3) for g, v in groups.items()} if name in ("math5",) else None,
                 "mean_out_chars": round(sum(map(len, outs)) / len(outs))}
    with open(f"{ROOT}/pilot/out2/hard_{name}__{tag}.jsonl", "w") as f:
        for it, o, k in zip(items, outs, ok):
            f.write(json.dumps({**it, "output": o, "correct": k}, ensure_ascii=False, default=str) + "\n")
    print(name, res[name], flush=True)
json.dump(res, open(f"{ROOT}/pilot/out2/hard__{tag}.json", "w"), indent=1)
