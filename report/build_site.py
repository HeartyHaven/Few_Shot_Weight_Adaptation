"""Build the data files of the report site from runs/, eval/, pilot/ and logs/ (everything the page shows live).

  python report/build_site.py            -> report/site/summary.json, report/site/runs/<run>.json, <run>_quizzes.json
Also writes report/site/_changed.json: files whose content differs from the last publish (report/.published.json).
After publishing, run  python report/build_site.py --mark-published  to record the published state.
"""
import glob, hashlib, json, os, sys, time

D = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(D); SITE = D + "/site"
os.makedirs(SITE + "/runs", exist_ok=True)
TEACHER = {"gpt-6-astra": "Astra 6", "gpt-6.2-astra": "Astra 6.2", "claude-fable-5-2": "Fable 5.2"}
STUDENT = {"gemma-3-1b-it": "Gemma 3 1B", "Qwen3-1.7B": "Qwen3 1.7B", "Qwen3-4B": "Qwen3 4B", "gemma-3-4b-it": "Gemma 3 4B", "Qwen3-8B": "Qwen3 8B"}
VOID = {"t2__gemma-3-1b-it__claude-fable-5-2__s1": "作废：教师代码触发显卡错误后沙盒无法恢复（框架缺陷，已修复），零编辑结束"}
CAPKEY = {"t2__gemma-3-1b-it__claude-fable-5-2__s0": "fable_s0", "t2__gemma-3-1b-it__gpt-6-astra__s0": "astra_s0", "t2__gemma-3-1b-it__gpt-6-astra__s1": "astra_s1"}


def rd(p):
    return open(p).read() if os.path.exists(p) else ""


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def pack_eval(res, task):
    c = lambda d: {"acc": round(d["acc"] * 100, 1), "lo": round(d["lo"] * 100, 1), "hi": round(d["hi"] * 100, 1), "n": d["n"]}
    out = {"sets": {}, "groups": {}}
    if task == "t2":
        for s in ("test_fresh", "test_bbh", "test_ood"):
            if s in res:
                out["sets"][s] = c(res[s]["free"]); out["groups"][s] = {g: c(v["free"]) for g, v in res[s]["by_group"].items()}
                out.setdefault("forced", {})[s] = c(res[s]["forced"])
    else:
        t = res["test"]; out["sets"] = {"taught": c(t["taught"]), "untaught": c(t["untaught"]), "all": c(t["free"])}
        out["groups"] = {g: c(v["free"]) for g, v in t["by_group"].items()}; out["mean_chars"] = t.get("mean_output_chars")
    r = res.get("retention")
    if r:
        out["retention"] = {"ppl": round(r["ppl_generic"], 2), "arc": round(r["arc_challenge"]["acc"] * 100, 1), "hs": round(r["hellaswag"]["acc"] * 100, 1), "arith": round(r["arithmetic"]["acc"] * 100, 1)}
    return out


def export_run(R, name, cfg):
    """Full per-run record for the archive view."""
    ev = jl(R + "/transcript.jsonl")
    if not ev:
        return 0
    t0 = ev[0]["t"]; notes = {}
    if os.path.exists(R + "/claude_stream.jsonl"):
        buf = []; n = 0
        for l in open(R + "/claude_stream.jsonl"):
            try:
                e = json.loads(l)
            except Exception:
                continue
            if e.get("type") != "assistant":
                continue
            for b in e["message"]["content"]:
                if b["type"] == "text" and b["text"].strip():
                    buf.append(b["text"].strip())
                elif b["type"] == "tool_use":
                    n += 1; notes[n] = "\n\n".join(buf); buf = []
    else:
        for e in ev:
            if e.get("event") == "assistant" and e.get("content"):
                notes[e["turn"] + 1] = e["content"]
    cap = {}
    ck = CAPKEY.get(name, name)
    if os.path.exists(f"{D}/captions/{ck}.json"):
        cap = {c["n"]: c for c in json.load(open(f"{D}/captions/{ck}.json"))}
    steps = []; k = 0
    for e in ev:
        if e.get("event") != "tool":
            continue
        k += 1; r = e.get("result") or {}
        s = {"n": k, "min": round((e["t"] - t0) / 60, 1), "tool": e["name"], "note": notes.get(k, ""), "cap": cap.get(k)}
        if e["name"] == "run_python":
            s.update(code=e.get("code", ""), out=r.get("stdout", ""), err=r.get("stderr", ""), ok=r.get("ok"), stat=r.get("stat"))
        else:
            s.update(code="", out=json.dumps(r, ensure_ascii=False), err="", ok="error" not in r, stat=None, why=e.get("note"))
        steps.append(s)
    led = jl(R + "/ledger.jsonl"); rb = [x for x in led if x["event"] == "rollback"]; removed = {e for r in rb for e in r["removed"]}
    edits = []
    for m in sorted(glob.glob(R + "/edits/*/meta.json")):
        e = json.load(open(m)); e.pop("shapes", None); e["rolled_back"] = e["edit_id"] in removed
        e["data_used"] = e["data_used"] if isinstance(e["data_used"], str) else json.dumps(e["data_used"], ensure_ascii=False)
        e["zh"] = EDIT_ZH.get(ck, {}).get(e["edit_id"]); edits.append(e)
    exams = [json.load(open(x)) for x in sorted(glob.glob(R + "/exams/*.json"))]
    out = {"name": name, "steps": steps, "edits": edits, "exams": exams, "rollbacks": rb, "val_looks": cfg.get("val_looks"),
           "syllabus": rd(R + "/work/syllabus.md"), "log": rd(R + "/work/curriculum_log.md"), "report": rd(R + "/final/report.md"),
           "system_prompt": rd(R + "/teacher_prompt.txt"), "syllabus_zh": rd(f"{D}/captions/{ck}_syllabus_zh.md"), "log_zh": rd(f"{D}/captions/{ck}_log_zh.md")}
    open(f"{SITE}/runs/{name}.json", "w").write(json.dumps(out, ensure_ascii=False).replace("\ufffd", "?"))
    quizzes = []
    for q in sorted(glob.glob(R + "/quizzes/*.jsonl")):
        rows = jl(q)
        if not rows:
            continue
        cut = lambda t: t if len(t) <= 6000 else t[:6000] + " …[截断]"
        quizzes.append({"quiz": rows[0]["quiz"], "label": rows[0]["label"], "skill": rows[0].get("skill"), "source": rows[0].get("source"), "after_edit": rows[0].get("after_edit"),
                        "uncommitted": rows[0].get("uncommitted"), "n": len(rows), "acc": round(sum(r["correct"] for r in rows) / len(rows), 4),
                        "items": [{"id": r.get("id"), "group": r.get("group"), "q": cut(r["question"]), "gold": r["answer"], "out": cut(r["output"]), "pred": r.get("pred"), "ok": r["correct"]} for r in rows]})
    open(f"{SITE}/runs/{name}_quizzes.json", "w").write(json.dumps(quizzes, ensure_ascii=False).replace("\ufffd", "?"))
    return len(cap)


def build():
    sys.path.insert(0, D)
    global EDIT_ZH
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("edit_zh", D + "/edit_zh.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); EDIT_ZH = mod.EDIT_ZH
    except Exception:
        EDIT_ZH = {}
    runs = []
    for R in sorted(glob.glob(ROOT + "/runs/*")):
        name = os.path.basename(R); cfg = json.load(open(R + "/config.json"))
        stag = cfg["student"].split("/")[-1]
        led = jl(R + "/ledger.jsonl"); rb = [x for x in led if x["event"] == "rollback"]; removed = {e for r in rb for e in r["removed"]}
        metas = [json.load(open(m)) for m in sorted(glob.glob(R + "/edits/*/meta.json"))]
        edits = [{"id": m["edit_id"], "accepted": m["accepted"], "rolled_back": m["edit_id"] in removed, "op": m["op_class"], "n": m["n_values_changed"]} for m in metas]
        exams = [[x["after_edit"], round(x["accuracy"] * 100, 1)] for x in (json.load(open(p)) for p in sorted(glob.glob(R + "/exams/*.json")))]
        calls = sum(1 for l in open(R + "/transcript.jsonl") if '"event": "tool"' in l) if os.path.exists(R + "/transcript.jsonl") else 0
        idle = round((time.time() - os.path.getmtime(R + "/transcript.jsonl")) / 60) if calls else None
        execs = [x for x in led if x["event"] == "exec"]
        fwd = sum(x.get("fwd_tokens", 0) for x in execs); bwd = sum(x.get("bwd_tokens", 0) for x in execs)
        fin = "finished" in cfg and cfg.get("finished")
        status = "void" if name in VOID else ("interrupted" if cfg.get("interrupted") else ("finished" if fin else "running"))
        note = VOID.get(name) or (("被中断：教师的订阅用量上限触发（" + cfg.get("interrupted", "")[-24:].replace("after ", "第 ").replace(" tool calls", " 步") + "），已提交的编辑保留") if cfg.get("interrupted") else "")
        r = {"name": name, "task": cfg["task"], "student": STUDENT.get(stag, stag), "student_tag": stag, "teacher": TEACHER.get(cfg["teacher"], cfg["teacher"]), "teacher_id": cfg["teacher"],
             "seed": cfg["seed"], "status": status, "note": note, "calls": calls, "idle_min": idle, "started": cfg.get("started"), "wall_min": round(cfg["wall_seconds"] / 60) if cfg.get("wall_seconds") else None,
             "edits": edits, "kept": sum(1 for e in edits if e["accepted"] and not e["rolled_back"]), "exams": exams, "fwd": fwd, "bwd": bwd,
             "mem": max([x.get("peak_mem_ratio", 0) for x in execs] or [0]), "restarts": sum(1 for x in led if x["event"] == "kernel_restart"), "eval": None, "values": None}
        ep = f"{ROOT}/eval/{name}.json"
        if os.path.exists(ep) and status != "void":
            e = json.load(open(ep)); r["eval"] = pack_eval(e["results"], cfg["task"]); r["values"] = e.get("values_changed")
            u = e.get("teacher_usage") or {}; r["usd"] = round(u["total_cost_usd"], 1) if u.get("total_cost_usd") else None
            r["teacher_out_tokens"] = u.get("completion_tokens") or (u.get("usage") or {}).get("output_tokens")
        r["caps"] = export_run(R, name, cfg) if status != "void" or calls else 0
        runs.append(r)
    base = {}
    for p in sorted(glob.glob(ROOT + "/eval/base__*.json")):
        e = json.load(open(p)); tag = e["student"].split("/")[-1]; key = e["task"] + "|" + tag
        if "__bf16" in p:
            key += "|bf16"
        base[key] = pack_eval(e["results"], e["task"]); base[key]["student"] = STUDENT.get(tag, tag)
    pilots = {}
    for p in sorted(glob.glob(ROOT + "/pilot/out2/summary__*.json")):
        pilots[os.path.basename(p)[9:-5]] = json.load(open(p))
    # math answers of the first two students were regraded with the tolerant extractor (see CHANGELOG)
    for tag, v in (("gemma-3-1b-it", 0.5133), ("Qwen3-1.7B", 0.7533)):
        if tag in pilots and pilots[tag].get("math", {}).get("acc") in (0.4333, 0.1667):
            pilots[tag]["math"]["acc"] = v; pilots[tag]["math"]["regraded"] = True
            pilots[tag]["code"]["stale"] = True
    queue = {"pending": [l.split() for l in rd(ROOT + "/logs/queue.txt").splitlines() if l.strip()], "launched": rd(ROOT + "/logs/queue.log").splitlines()[-12:],
             "big": [l for l in rd(ROOT + "/logs/bigpilot.log").splitlines() if "start" in l or "done" in l or "FAILED" in l][-10:], "big_pending": [l.split()[0] for l in rd(ROOT + "/logs/bigpilot.txt").splitlines() if l.strip()]}
    launched = {" ".join(l.split()[2:6]) for l in queue["launched"] if " launch " in l}
    summary = {"generated": time.strftime("%Y-%m-%d %H:%M"), "runs": runs, "base": base, "pilots": pilots, "queue": queue}
    json.dump(summary, open(SITE + "/summary.json", "w"), ensure_ascii=False)
    pub = json.load(open(D + "/.published.json")) if os.path.exists(D + "/.published.json") else {}
    changed = []
    for f in sorted(glob.glob(SITE + "/summary.json") + glob.glob(SITE + "/runs/*.json")):
        rel = os.path.relpath(f, D); h = hashlib.md5(open(f, "rb").read()).hexdigest()
        if rel == "site/summary.json":
            h = hashlib.md5(json.dumps({k: v for k, v in summary.items() if k != "generated"}, sort_keys=True).encode()).hexdigest()
        if pub.get(rel) != h:
            changed.append([rel, h])
    json.dump(changed, open(D + "/.changed.json", "w"))
    page = open(D + "/tdwa_report.html").read().replace("<title>", "</head><body><title>", 1)      # standalone copy for the local server
    open(D + "/index.html", "w").write('<!doctype html>\n<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                                       '<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n' + page + "\n</body></html>\n")
    print(summary["generated"], len(runs), "runs;", "changed files:", len(changed), "size MB:", round(sum(os.path.getsize(D + "/" + c[0]) for c in changed) / 1e6, 1))
    return changed


if __name__ == "__main__":
    if "--mark-published" in sys.argv:
        pub = json.load(open(D + "/.published.json")) if os.path.exists(D + "/.published.json") else {}
        for rel, h in json.load(open(D + "/.changed.json")):
            pub[rel] = h
        json.dump(pub, open(D + "/.published.json", "w")); print("marked", len(pub))
    else:
        for rel, h in build():
            print("  ", rel)
