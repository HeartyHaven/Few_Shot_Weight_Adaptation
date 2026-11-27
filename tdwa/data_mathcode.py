"""Splits for task T4 (MATH Level 5: number theory, counting & probability) and T5 (CRUXEval-O output prediction).

  data/<t>/dev.jsonl         100 items, visible to the teacher (with answers)
  _secret/<t>/val.jsonl      150 items, hidden (aggregate scores only)
  _secret/<t>/test.jsonl     hidden test, evaluator only

T4 (experiments.md): dev and val from the MATH *train* split of the two subjects, Level 5 only, half per subject;
   test = all Level 5 items of the two subjects in the MATH *test* split (154 number theory + 123 counting & probability).
   group = subject. Answer = last \\boxed{} of the reference solution. The WP0 pilot used number-theory test Levels 1-3,
   so no test item here has been seen.
T5: CRUXEval has only its 800-item official set. experiments.md plans a generated training set (as in the CRUXEval paper);
   until that exists, the 800 items are split 100 dev / 150 val / 550 test (seeded shuffle). group = "all".
   The WP0 pilot used MBPP, not CRUXEval, so no test item here has been seen.
Usage: python -m tdwa.data_mathcode
"""
import json, os, random

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBJECTS = {"number_theory": "Number Theory", "counting_and_probability": "Counting & Probability"}


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def math_items(split, level5=True):
    out = []
    for sub, name in SUBJECTS.items():
        df = pd.read_parquet(hf_hub_download("EleutherAI/hendrycks_math", f"{sub}/{split}-00000-of-00001.parquet", repo_type="dataset"))
        for i, r in df.iterrows():
            if level5 and r["level"] != "Level 5":
                continue
            ans = last_boxed(r["solution"])
            if ans is None:
                continue
            out.append({"id": f"{split}-{sub}-{i}", "task": "t4", "group": name, "question": r["problem"], "answer": ans})
    return out


def build_t4():
    rng = random.Random(4404)
    train = math_items("train"); test = math_items("test")
    dev, val = [], []
    for name in SUBJECTS.values():
        pool = [x for x in train if x["group"] == name]; rng.shuffle(pool)
        dev += pool[:50]; val += pool[50:125]
    return {"dev": dev}, {"val": val, "test": test}


def build_t5():
    path = hf_hub_download("cruxeval-org/cruxeval", "test.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(path)]
    items = []
    for r in rows:
        q = ("Here is a Python function and an input. Predict the exact value that the function returns for this input.\n\n"
             f"```python\n{r['code'].rstrip()}\n```\n\nInput: f({r['input']})")
        items.append({"id": r["id"], "task": "t5", "group": "all", "question": q, "answer": r["output"]})
    random.Random(5505).shuffle(items)
    return {"dev": items[:100]}, {"val": items[100:250], "test": items[250:]}


if __name__ == "__main__":
    from tdwa.data import write
    for t, build in (("t4", build_t4), ("t5", build_t5)):
        pub, sec = build()
        for k, v in pub.items():
            write(f"{ROOT}/data/{t}/{k}.jsonl", v); print(t, k, len(v), {g: sum(x['group'] == g for x in v) for g in sorted({x['group'] for x in v})})
        for k, v in sec.items():
            write(f"{ROOT}/_secret/{t}/{k}.jsonl", v); print(t, k, len(v), {g: sum(x['group'] == g for x in v) for g in sorted({x['group'] for x in v})}, "(secret)")
