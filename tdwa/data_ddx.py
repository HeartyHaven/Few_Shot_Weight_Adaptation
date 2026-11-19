"""Splits for task T3: DDXPlus differential diagnosis (5 options).

  data/t3/dev.jsonl          100 items, visible to the teacher (with answers)       <- DDXPlus validate.csv
  _secret/t3/val.jsonl       150 items, hidden (aggregate scores only)              <- DDXPlus validate.csv (disjoint rows)
  _secret/t3/test.jsonl      600 items, hidden, evaluator only                      <- DDXPlus test.csv

Item format = pilot/scripts/baseline2.py (findings from the patient interview, the true pathology plus the 4 most likely
other diseases of the case's own differential, shuffled into options A-E). Differences from the pilot:
  * dev/val come from validate.csv, test from test.csv; the 150 pilot items (test.csv sample, random_state=11) are excluded
    from the test split, so no test item has been read by anyone.
  * pathologies are sampled round-robin so that every pathology with enough cases appears in each split
    (49 pathologies: dev gets about 2 per pathology); `group` = pathology name.
Usage: python -m tdwa.data_ddx
"""
import ast, json, os, random

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "aai530-group6/ddxplus"


def render(r, ev, rng):
    diff = [d for d, p in ast.literal_eval(r["DIFFERENTIAL_DIAGNOSIS"]) if d != r["PATHOLOGY"]][:4]
    if len(diff) < 4:
        return None
    lines = {}
    for code in ast.literal_eval(r["EVIDENCES"]):
        k, _, v = code.partition("_@_"); e = ev[k]; q = e["question_en"]
        if not v:
            lines.setdefault(q, []).append("yes")
        else:
            vm = e.get("value_meaning", {}).get(v, {}).get("en", v); lines.setdefault(q, []).append(str(vm))
    opts = diff + [r["PATHOLOGY"]]; rng.shuffle(opts)
    text = (f"Patient: {r['AGE']} years old, sex {r['SEX']}.\nFindings from the interview:\n" + "\n".join(f"- {q} {', '.join(a)}" for q, a in lines.items()) +
            "\n\nWhich diagnosis is most likely?\nOptions:\n" + "\n".join(f"({'ABCDE'[i]}) {o}" for i, o in enumerate(opts)))
    return {"task": "t3", "group": r["PATHOLOGY"], "question": text, "answer": "(" + "ABCDE"[opts.index(r["PATHOLOGY"])] + ")",
            "n_options": 5, "options": opts, "initial_evidence": r["INITIAL_EVIDENCE"]}


def balanced(df, n, seed, ev, prefix):
    """Round-robin over pathologies (shuffled order), random case within each pathology."""
    rng = random.Random(seed); pools = {}
    for i, r in df.sample(frac=1.0, random_state=seed).iterrows():
        pools.setdefault(r["PATHOLOGY"], []).append((i, r))
    names = sorted(pools); rng.shuffle(names); out = []; used = set()
    while len(out) < n and any(pools.values()):
        for p in names:
            while pools[p] and len(out) < n:
                i, r = pools[p].pop()
                it = render(r, ev, rng)
                if it:
                    it["id"] = f"{prefix}-{i}"; out.append(it); used.add(i); break
    return out, used


def build():
    ev = json.load(open(hf_hub_download(REPO, "release_evidences.json", repo_type="dataset")))
    va = pd.read_csv(hf_hub_download(REPO, "validate.csv", repo_type="dataset"))
    te = pd.read_csv(hf_hub_download(REPO, "test.csv", repo_type="dataset"))
    pilot = set(te.sample(n=4000, random_state=11).index[:400])     # superset of the 150 pilot items (pilot stopped at 150 valid)
    dev, used = balanced(va, 100, 6606, ev, "dev")
    val, _ = balanced(va.drop(index=list(used)), 150, 7707, ev, "val")
    test, _ = balanced(te.drop(index=list(pilot)), 600, 8808, ev, "test")
    return {"dev": dev}, {"val": val, "test": test}


if __name__ == "__main__":
    from tdwa.data import write
    pub, sec = build()
    for k, v in pub.items():
        write(f"{ROOT}/data/t3/{k}.jsonl", v); print("t3", k, len(v), "groups", len({x['group'] for x in v}))
    for k, v in sec.items():
        write(f"{ROOT}/_secret/t3/{k}.jsonl", v); print("t3", k, len(v), "groups", len({x['group'] for x in v}), "(secret)")
