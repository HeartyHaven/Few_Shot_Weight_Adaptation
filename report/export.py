"""Export finished teaching runs into JSON files the report page loads."""
import json, glob, os, sys, ast
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {"fable_s0": "t2__gemma-3-1b-it__claude-fable-5-2__s0", "astra_s0": "t2__gemma-3-1b-it__gpt-6-astra__s0", "astra_s1": "t2__gemma-3-1b-it__gpt-6-astra__s1",
        "fable_s2": "t2__gemma-3-1b-it__claude-fable-5-2__s2", "astra_s2": "t2__gemma-3-1b-it__gpt-6-astra__s2",
        "t1_astra62_s0": "t1__gemma-3-1b-it__gpt-6.2-astra__s0", "t1_fable_s0": "t1__gemma-3-1b-it__claude-fable-5-2__s0",
        "t1_astra6_s0": "t1__gemma-3-1b-it__gpt-6-astra__s0", "t1_astra6_s1": "t1__gemma-3-1b-it__gpt-6-astra__s1"}
def rd(p):
    return open(p).read() if os.path.exists(p) else ""
for key, name in RUNS.items():
    R = os.path.join(ROOT, "runs", name); ev = [json.loads(l) for l in open(R + "/transcript.jsonl")]
    t0 = ev[0]["t"]; notes = {}
    if os.path.exists(R + "/claude_stream.jsonl"):      # Fable: narration/thinking precede each tool_use
        buf = []; n = 0
        for l in open(R + "/claude_stream.jsonl"):
            e = json.loads(l)
            if e.get("type") != "assistant": continue
            for b in e["message"]["content"]:
                if b["type"] == "text" and b["text"].strip(): buf.append(b["text"].strip())
                elif b["type"] == "thinking" and (b.get("thinking") or "").strip(): buf.append("[thinking] " + b["thinking"].strip())
                elif b["type"] == "tool_use": n += 1; notes[n] = "\n\n".join(buf); buf = []
    else:
        for e in ev:
            if e.get("event") == "assistant" and e.get("content"): notes[e["turn"] + 1] = e["content"]
    steps = []; k = 0
    for e in ev:
        if e.get("event") == "tool":
            k += 1; r = e.get("result") or {}
            s = {"n": k, "min": round((e["t"] - t0) / 60, 1), "tool": e["name"], "note": notes.get(k, "")}
            if e["name"] == "run_python":
                s.update(code=e.get("code", ""), out=r.get("stdout", ""), err=r.get("stderr", ""), ok=r.get("ok"), stat=r.get("stat"))
            else:
                s.update(code="", out=json.dumps(r, ensure_ascii=False), err="", ok="error" not in r, stat=None, why=e.get("note"))
            steps.append(s)
    quizzes = []
    for q in sorted(glob.glob(R + "/quizzes/*.jsonl")):
        rows = [json.loads(l) for l in open(q)]
        if not rows: continue
        quizzes.append({"quiz": rows[0]["quiz"], "label": rows[0]["label"], "skill": rows[0].get("skill"), "source": rows[0].get("source"), "after_edit": rows[0].get("after_edit"),
                        "uncommitted": rows[0].get("uncommitted"), "n": len(rows), "acc": round(sum(r["correct"] for r in rows) / len(rows), 4),
                        "items": [{"id": r.get("id"), "group": r.get("group"), "q": r["question"], "gold": r["answer"], "out": r["output"], "pred": r.get("pred"), "ok": r["correct"]} for r in rows]})
    edits = [json.load(open(m)) for m in sorted(glob.glob(R + "/edits/*/meta.json"))]
    for m in edits: m.pop("shapes", None)
    exams = [json.load(open(x)) for x in sorted(glob.glob(R + "/exams/*.json"))]
    led = [json.loads(l) for l in open(R + "/ledger.jsonl")]
    out = {"name": name, "config": json.load(open(R + "/config.json")), "steps": steps, "edits": edits, "exams": exams,
           "rollbacks": [x for x in led if x["event"] == "rollback"], "syllabus": rd(R + "/work/syllabus.md"), "log": rd(R + "/work/curriculum_log.md"),
           "report": rd(R + "/final/report.md"), "system_prompt": rd(R + "/teacher_prompt.txt")}
    json.dump(out, open(os.path.join(ROOT, "report", "data", key + ".json"), "w"), ensure_ascii=False)
    json.dump(quizzes, open(os.path.join(ROOT, "report", "data", key + "_quizzes.json"), "w"), ensure_ascii=False)
    # compact file for caption writers
    with open(os.path.join(ROOT, "report", "data", key + "_steps.txt"), "w") as f:
        for s in steps:
            f.write(f"\n######## STEP {s['n']} | minute {s['min']} | tool {s['tool']}\n--- teacher note ---\n{s['note'][:1500]}\n--- code ---\n{s['code'][:6000]}\n--- output ---\n{(s['out'] or '')[:3500]}\n--- stderr ---\n{(s['err'] or '')[:800]}\n")
    print(key, len(steps), "steps", len(quizzes), "quizzes", len(edits), "edits", sum(1 for s in steps if s["note"]), "notes")
