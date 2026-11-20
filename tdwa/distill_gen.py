"""Baseline B3 data: the teacher (Astra 6.2 through headless Codex) writes worked solutions, which then train a LoRA.

Same data allowance as a teaching run: the 100 dev items plus at most 300 teacher-made items.
  t1: solutions for the 100 dev cases (gold answer given) + 300 NEW patient notes the teacher writes for the 8 taught
      calculators (~38 each), each with question, worked solution and answer;
  t2: solutions for the 100 dev items + 300 fresh generator items (seed 7171, 3/5/7 objects 20/40/40%), gold given.
Only solutions whose final answer matches the gold answer are kept (teacher-written t1 cases: the teacher's own answer is
the label, and the solution must end in a parseable "Final answer:").
Codex runs in the same isolation as tdwa/codex_teach.py (mount namespace hiding the project; throw-away CODEX_HOME).
Usage: python -m tdwa.distill_gen --task t1 [--model gpt-6.2-astra] [--workers 6]
Output: baselines/B3_data/<task>.jsonl (+ usage.json); train with  python -m tdwa.baseline_lora --task t1 --data distill ...
"""
import argparse, concurrent.futures as cf, json, os, random, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tdwa import grade as G  # noqa: E402
from tdwa import data as D  # noqa: E402
from tdwa.codex_teach import HIDE, HOME  # noqa: E402

CALCS = {45: "CURB-65 Score for Pneumonia Severity", 51: "SIRS Criteria", 48: "PERC Rule for Pulmonary Embolism",
         8: "Wells' Criteria for Pulmonary Embolism", 20: "Centor Score (Modified/McIsaac) for Strep Pharyngitis",
         15: "Child-Pugh Score for Cirrhosis Mortality", 25: "HAS-BLED Score for Major Bleeding Risk",
         4: "CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk"}
SOLVE = """You are writing worked solutions that will be used to fine-tune a small language model. For each problem below,
write the solution the small model should learn to produce: clear, step by step, short enough (well under 400 words),
and ending with a final line exactly of the form `Final answer: {fmt}`. The correct answer is given to you; your
solution must reach it by correct reasoning (do not mention that the answer was given).
Return ONLY a JSON array, one object per problem, in order: [{{"id": "...", "solution": "..."}}, ...]. No other text.

"""
CREATE = """You are creating training data for a small language model that must learn the {calc}.
Write {n} NEW, realistic and varied patient notes (each 120-300 words, like case reports: history, exam, vitals, labs as
relevant; vary ages, sexes, and which criteria are present/absent/borderline; some values near thresholds). For each,
pose the question "What is the patient's {calc}?" (phrased naturally), and write the worked solution the small model
should learn: list the criteria, extract each value from the note, score it, sum, and end with a final line exactly
`Final answer: <number>`. Make sure the arithmetic is right.
Return ONLY a JSON array: [{{"note": "...", "question": "...", "solution": "...", "answer": <number>}}, ...]. No other text."""


def codex(prompt, model, effort, workdir):
    """One isolated `codex exec` call; returns (last message text, usage dict)."""
    chome = os.path.join(workdir, "codex_home"); os.makedirs(chome, exist_ok=True)
    shutil.copy(os.path.join(HOME, ".codex", "auth.json"), os.path.join(chome, "auth.json")); open(os.path.join(chome, "config.toml"), "w").write("")
    empty = os.path.join(workdir, "empty"); os.makedirs(empty, exist_ok=True); cwd = os.path.join(workdir, "cwd"); os.makedirs(cwd, exist_ok=True)
    out = os.path.join(workdir, "last.txt")
    cmd = ["codex", "exec", "-m", model, "--skip-git-repo-check", "--sandbox", "read-only", "--json", "-C", cwd, "-o", out,
           "-c", f"model_reasoning_effort={effort}", "-c", "tools.web_search=false", "-c", "web_search=\"disabled\"", prompt]
    mounts = " && ".join(f"mount --bind {empty} {h}" for h in HIDE if os.path.isdir(h))
    env = {k: v for k, v in os.environ.items() if not any(s in k for s in ("HF_", "CLIPROXY", "OPENAI_API", "ANTHROPIC", "TDWA"))}
    env.update(CODEX_HOME=chome)
    p = subprocess.run(["unshare", "-rm", "sh", "-c", mounts + ' && exec "$@"', "sh"] + cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=3600)
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_output_tokens": 0}
    for line in p.stdout.splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "turn.completed":
            for k in usage:
                usage[k] += (e.get("usage") or {}).get(k) or 0
    shutil.rmtree(chome, ignore_errors=True)
    return (open(out).read() if os.path.exists(out) else ""), usage


def parse_json(text):
    s, e = text.find("["), text.rfind("]")
    return json.loads(text[s:e + 1]) if s >= 0 and e > s else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--model", default="gpt-6.2-astra"); ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--style", default="free", choices=["free", "table"], help="table (t2 only): require the full state table after every swap")
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, "baselines", "B3_data"); os.makedirs(out_dir, exist_ok=True)
    dev = [json.loads(l) for l in open(os.path.join(ROOT, "data", a.task, "dev.jsonl"))]
    jobs = []
    fmt = "<number>" if a.task == "t1" else "(X)"
    style = ("" if a.style == "free" else
             "REQUIRED FORMAT: first write the initial assignment of every person, then for EVERY swap in order write the full "
             "assignment of every person after that swap (one line per person), then state what the asked person has and match it "
             "to the option letter. Do not skip swaps and do not trace backwards.\n\n")
    if a.task == "t2":
        rng = random.Random(7171); own = []
        for n, c in ((3, 60), (5, 120), (7, 120)):
            own += [D.make_item(rng, n, None, f"own-{n}-{i}") for i in range(c)]
        items = dev + own
    else:
        items = dev
    for i in range(0, len(items), a.batch):
        b = items[i:i + a.batch]
        body = "\n\n".join(f"### Problem id={it['id']}\n{G.format_prompt(it)}\nCorrect answer: {it['answer']}" for it in b)
        jobs.append(("solve", b, SOLVE.format(fmt=fmt) + style + body))
    if a.task == "t1":                       # 300 teacher-written cases: 38 per calculator (8 x 38 = 304, trimmed to 300)
        per = {cid: 38 for cid in CALCS}
        for cid, n in per.items():
            for k in range(0, n, 5):
                jobs.append(("create", cid, CREATE.format(calc=CALCS[cid], n=min(5, n - k))))
    print(len(jobs), "codex calls", flush=True)
    rows, usage, t0 = [], {}, time.time()

    def work(j):
        wd = tempfile.mkdtemp(prefix="tdwa_b3_")
        try:
            text, u = codex(j[2], a.model, a.effort, wd)
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        return j, text, u

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for j, text, u in ex.map(work, jobs):
            for k, v in u.items():
                usage[k] = usage.get(k, 0) + v
            try:
                got = parse_json(text)
            except Exception:
                got = []
            if j[0] == "solve":
                byid = {it["id"]: it for it in j[1]}
                for g in got:
                    it = byid.get(str(g.get("id")))
                    if not it or not g.get("solution"):
                        continue
                    ok = G.grade(it, g["solution"])["correct"]
                    rows.append({"id": it["id"], "source": "dev" if it in dev else "generated", "task": a.task, "question": it["question"],
                                 "answer": it["answer"], "solution": g["solution"], "kept": ok, **({"lower": it.get("lower"), "upper": it.get("upper")} if a.task == "t1" else {})})
            else:
                for k, g in enumerate(got):
                    sol, ans = g.get("solution") or "", g.get("answer")
                    pred = G.parse("t1", sol)
                    ok = bool(g.get("note") and g.get("question")) and pred is not None and ans is not None and abs(float(pred) - float(ans)) < 1e-6
                    rows.append({"id": f"own-{j[1]}-{len(rows)}", "source": "teacher_written", "task": "t1", "group": CALCS[j[1]],
                                 "question": f"Patient note:\n{g.get('note', '')}\n\nQuestion: {g.get('question', '')}", "answer": ans, "solution": sol, "kept": ok})
            print(f"{len(rows)} rows, {sum(r['kept'] for r in rows)} kept, {round(time.time() - t0)} s, usage {usage}", flush=True)
    with open(os.path.join(out_dir, f"{a.task}{'' if a.style == 'free' else '_' + a.style}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"task": a.task, "model": a.model, "effort": a.effort, "calls": len(jobs), "rows": len(rows), "kept": sum(r["kept"] for r in rows),
               "usage": usage, "seconds": round(time.time() - t0)}, open(os.path.join(out_dir, f"{a.task}{'' if a.style == 'free' else '_' + a.style}_usage.json"), "w"), indent=1)
    print("done", len(rows), sum(r["kept"] for r in rows))


if __name__ == "__main__":
    main()
