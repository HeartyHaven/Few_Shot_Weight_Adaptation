"""Re-grade saved hidden-test outputs with the current tdwa/grade.py, without generating again.

  python -m tdwa.regrade <eval name> [...]     e.g. base__t4__Qwen3-1.7B

Reads _secret/eval/<name>/<split>.jsonl (the outputs tdwa.evaluate saved) and the items in _secret/<task>/<split>.jsonl,
re-grades the free-generation answers, rewrites those rows and the free-generation fields of eval/<name>.json
(free, no_final_answer_rate, by_group[*].free, taught/untaught). Forced-choice and retention results are kept.
The old file is kept as eval/<name>.json.pre_regrade.
"""
import json, os, shutil, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tdwa import grade as G  # noqa: E402


def ci(c):   # same as tdwa/evaluate.py
    a = np.array(c, dtype=float); rng = np.random.default_rng(0)
    bs = rng.choice(a, size=(1000, len(a)), replace=True).mean(1)
    return {"acc": round(float(a.mean()), 4), "lo": round(float(np.percentile(bs, 2.5)), 4), "hi": round(float(np.percentile(bs, 97.5)), 4), "n": len(a)}


def regrade(name):
    path = os.path.join(ROOT, "eval", name + ".json"); ev = json.load(open(path)); task = ev["task"]
    if not os.path.exists(path + ".pre_regrade"):
        shutil.copy(path, path + ".pre_regrade")
    for split, d in ev["results"].items():
        rp = os.path.join(ROOT, "_secret", "eval", name, split + ".jsonl")
        if split == "retention" or not os.path.exists(rp):
            continue
        items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "_secret", task, split + ".jsonl"))}
        rows = [json.loads(l) for l in open(rp)]
        for r in rows:
            r.update(G.grade(items[r["id"]], r["output"]))
        with open(rp, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        old = d["free"]["acc"]
        d["free"] = ci([r["correct"] for r in rows])
        d["no_final_answer_rate"] = round(sum(r["pred"] is None for r in rows) / len(rows), 4)
        keyf = (lambda r: ("taught/" if r["taught"] else "untaught/") + r["group"]) if task == "t1" else (lambda r: r["group"])
        for g in d.get("by_group", {}):
            d["by_group"][g]["free"] = ci([r["correct"] for r in rows if keyf(r) == g])
        if task == "t1":
            for k, flag in (("taught", True), ("untaught", False)):
                d[k] = ci([r["correct"] for r in rows if r["taught"] == flag])
        print(f"{name} {split}: {old} -> {d['free']['acc']} (no answer {d['no_final_answer_rate']})")
    ev["regraded"] = "tdwa/regrade.py with the grade.py of " + __import__("time").strftime("%Y-%m-%d %H:%M")
    json.dump(ev, open(path, "w"), indent=1, default=str)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        regrade(n)
