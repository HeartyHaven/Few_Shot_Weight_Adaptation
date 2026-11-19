"""Aggregate the control groups (B0 vLLM base, B1 prompt, B2/B4 LoRA, B5 prompt baking) and extra baselines into
report/site/baselines.json, rendered by the "对照组" tab of report/tdwa_report.html.

Reads eval/*.json written by tdwa/evaluate_vllm.py (engine == "vllm"), baselines/*/config.json, pilot/out2/summary__*.json
(new-task pilots measured with vLLM) and the local scheduler state. Aggregates only: no test items are exported.
Usage: python report/build_baselines.py      (prints "changed" when the JSON content differs from the last build)
"""
import glob, hashlib, json, os, re, socket, statistics as S, time

D = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(D); SITE = os.path.join(D, "site")
STUDENT = {"gemma-3-1b-it": "Gemma 3 1B", "Qwen3-1.7B": "Qwen3 1.7B", "Qwen3-4B": "Qwen3 4B", "gemma-3-4b-it": "Gemma 3 4B",
           "Qwen3-8B": "Qwen3 8B", "gemma-3-12b-it": "Gemma 3 12B", "Qwen3-14B": "Qwen3 14B", "gemma-3-27b-it": "Gemma 3 27B"}
METHOD = {"B0": "未编辑", "B1": "提示：把解题流程写进题目前", "B2": "LoRA：只用 100 道真题", "B4": "LoRA：全量训练数据",
          "B5": "提示烘焙：把流程蒸进 LoRA"}
SETS = {"t1": ["taught", "untaught", "all"], "t2": ["test_fresh", "test_bbh", "test_ood"], "t3": ["test"]}


def sets_of(task, res):
    if task == "t1":
        t = res["test"]; return {"taught": t["taught"], "untaught": t["untaught"], "all": t["free"]}
    out = {k: res[k]["free"] for k in SETS[task] if k in res}
    if task == "t3" and "forced" in res.get("test", {}):
        out["test_forced"] = res["test"]["forced"]
    return out


def pct(x):
    return {"acc": round(x["acc"] * 100, 1), "lo": round(x["lo"] * 100, 1), "hi": round(x["hi"] * 100, 1), "n": x["n"]}


def ret(r):
    return {"ppl": round(r["ppl_generic"], 2), "arc": round(r["arc_challenge"]["acc"] * 100, 1), "hs": round(r["hellaswag"]["acc"] * 100, 1),
            "arith": round(r["arithmetic"]["acc"] * 100, 1)}


def build():
    groups = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "eval", "*.json"))):
        e = json.load(open(p))
        if e.get("engine") != "vllm":
            continue
        name = os.path.basename(p)[:-5]
        if name.endswith("__adapter"):            # consistency check run (adapter vs merged), not a separate result
            continue
        tag = e["student"].split("/")[-1]; task = e["task"]
        m = re.match(r"(B\d)__", name); method = m.group(1) if m else "B0"
        if method == "B0" and e.get("b1_prefix"):
            method = "B1"
        seed = int(re.search(r"__s(\d+)$", name).group(1)) if re.search(r"__s(\d+)$", name) else None
        groups.setdefault((task, tag, method), []).append((seed, name, e))
    rows = []
    for (task, tag, method), runs in sorted(groups.items()):
        per = [sets_of(task, e["results"]) for _, _, e in runs]; rets = [ret(e["results"]["retention"]) for _, _, e in runs]
        sets = {}
        for k in per[0]:
            accs = [pct(x[k])["acc"] for x in per if k in x]
            first = pct(per[0][k])
            sets[k] = {"acc": round(S.mean(accs), 1), "sd": round(S.stdev(accs), 1) if len(accs) > 1 else None,
                       "lo": first["lo"] if len(accs) == 1 else None, "hi": first["hi"] if len(accs) == 1 else None, "n": first["n"], "seeds": len(accs)}
        r = {"task": task, "student_tag": tag, "student": STUDENT.get(tag, tag), "method": method, "label": METHOD[method], "seeds": len(runs),
             "runs": [n for _, n, _ in runs], "sets": sets, "retention": {k: round(S.mean(x[k] for x in rets), 2 if k == "ppl" else 1) for k in rets[0]}}
        cfgs = [json.load(open(c)) for c in (os.path.join(ROOT, "baselines", n, "config.json") for _, n, _ in runs) if os.path.exists(c)]
        if cfgs:
            c = cfgs[0]
            r["train"] = {"examples": c.get("n_examples"), "epochs": c.get("epochs"), "steps": c.get("steps"),
                          "train_tokens": c.get("train_tokens") or c.get("forward_tokens"), "minutes": round(S.mean(x.get("train_seconds", 0) for x in cfgs) / 60, 1)}
        rows.append(r)
    pilots = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "pilot", "out2", "summary__*.json"))):
        d = json.load(open(p))
        if any(isinstance(v, dict) and v.get("engine") == "vllm-bf16" for v in d.values()):
            pilots[os.path.basename(p)[9:-5]] = {k: {"acc": round(v["acc"] * 100, 1), "n": v["n"]} for k, v in d.items()}
    host = socket.gethostname(); st = os.path.join(ROOT, "logs", f"sched_{host}.state.json"); jobs = os.path.join(ROOT, "logs", f"jobs_{host}.jsonl")
    state = json.load(open(st)) if os.path.exists(st) else {}
    pending = [json.loads(l)["id"] for l in open(jobs) if l.strip()] if os.path.exists(jobs) else []
    queue = {"running": sorted(k for k, v in state.items() if v and v.get("status") == "running"),
             "waiting": [j for j in pending if j not in state]}
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "host": host, "engine": "vLLM 0.11，bf16，贪心解码", "rows": rows, "pilots": pilots, "queue": queue}
    os.makedirs(SITE, exist_ok=True); path = os.path.join(SITE, "baselines.json")
    h = lambda d: hashlib.md5(json.dumps({k: v for k, v in d.items() if k != "generated"}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    old = json.load(open(path)) if os.path.exists(path) else {}
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=0)
    print("changed" if not old or h(old) != h(out) else "unchanged", len(rows), "rows", len(pilots), "pilots")


if __name__ == "__main__":
    build()
