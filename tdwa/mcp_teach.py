"""Teaching run driven by a headless Claude Code agent (`claude -p`) through a minimal MCP stdio server.

Driver:  python -m tdwa.mcp_teach run --task t2 --student google/gemma-3-1b-it --teacher claude-fable-5-2 --seed 0
The driver starts `claude -p` in an empty directory with every built-in tool disabled; the only tools the
teacher has are the three served here (run_python, official_validation, finish), identical to tdwa.teach.
Server (spawned by claude):  python -m tdwa.mcp_teach serve <run_dir>
"""
import argparse, json, os, subprocess, sys, tempfile, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tdwa import grade as G  # noqa: E402
from tdwa.teach import HF, SYSTEM, TASK_TEXT, TOOLS, Sandbox  # noqa: E402


def serve(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    name = os.path.basename(run_dir)
    tlog = open(os.path.join(run_dir, "transcript.jsonl"), "a")
    lock = threading.Lock()

    def log(**kw):
        tlog.write(json.dumps({"t": time.time(), **kw}, ensure_ascii=False, default=str) + "\n"); tlog.flush()

    state = {"sb": None, "turns": 0, "looks": 0, "done": False, "t0": time.time()}
    ready = threading.Event()

    def boot():
        sb = Sandbox(cfg, run_dir, os.path.join(HF, "models--" + cfg["student"].replace("/", "--")), cfg["gpu"], cfg["exec_timeout"])
        state["sb"] = sb; log(event="sandbox_ready", ready=sb.ready, hidden=sb.hidden); ready.set()
    threading.Thread(target=boot, daemon=True).start()
    val = [json.loads(l) for l in open(os.path.join(ROOT, "_secret", cfg["task"], "val.jsonl"))]
    secret_dir = os.path.join(ROOT, "_secret", "exams", name); os.makedirs(secret_dir, exist_ok=True)

    def finalize(report):
        if state["done"]:
            return
        state["done"] = True
        fin = state["sb"].rpc({"cmd": "finish"}, timeout=1800); state["sb"].close()
        os.makedirs(os.path.join(run_dir, "final"), exist_ok=True)
        open(os.path.join(run_dir, "final", "report.md"), "w").write(report or "(the teacher did not submit a final report)")
        cfg.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), wall_seconds=round(time.time() - state["t0"]), turns=state["turns"],
                   validation_looks=state["looks"], finish=fin, report_submitted=report is not None)
        json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1); log(event="end", **cfg)

    def call(tool, args):
        ready.wait(1200); sb = state["sb"]
        if state["done"]:
            return "the run has ended"
        state["turns"] += 1; turn = state["turns"]
        left = cfg["max_turns"] - turn
        note = f"\n[harness: {left} tool calls remain]" if left <= 15 else ""
        if left < 0 and tool != "finish":
            return "tool-call budget exhausted; call finish now with your final report"
        if turn == 1 and tool == "run_python":
            pass
        if tool == "run_python":
            r = sb.rpc({"code": args.get("code", ""), "call": turn})
            log(event="tool", turn=turn, name="run_python", code=args.get("code", ""), result=r)
            out = (r.get("stdout") or "") + (("\n[stderr]\n" + r["stderr"]) if r.get("stderr") else "")
            return (out.strip() or "(no output)") + f"\n[{json.dumps(r.get('stat', {}))}]" + note
        if tool == "official_validation":
            if state["looks"] >= cfg["val_looks"]:
                return json.dumps({"error": "no validation looks left"})
            r = sb.rpc({"cmd": "generate", "prompts": [G.format_prompt(it) for it in val], "max_new_tokens": cfg.get("val_max_new_tokens", 2048)}, timeout=7200)
            if not r.get("ok"):
                res = {"error": r.get("error") or r.get("stderr")}
            else:
                state["looks"] += 1; k = state["looks"]
                rows = [{"id": it["id"], "group": it["group"], "answer": it["answer"], "output": o, **G.grade(it, o)} for it, o in zip(val, r["outputs"])]
                by = {}
                for x in rows:
                    by.setdefault(x["group"], []).append(x["correct"])
                res = {"look": k, "looks_left": cfg["val_looks"] - k, "after_edit": r["after_edit"], "n": len(rows),
                       "accuracy": round(sum(x["correct"] for x in rows) / len(rows), 4),
                       "by_group": {g: round(sum(v) / len(v), 4) for g, v in sorted(by.items())},
                       "no_final_answer_rate": round(sum(x["pred"] is None for x in rows) / len(rows), 4)}
                json.dump({**res, "note": args.get("note")}, open(os.path.join(run_dir, "exams", f"{k:02d}.json"), "w"), indent=1)
                with open(os.path.join(secret_dir, f"{k:02d}.jsonl"), "w") as f:
                    for x in rows:
                        f.write(json.dumps(x, ensure_ascii=False) + "\n")
            log(event="tool", turn=turn, name="official_validation", note=args.get("note"), result=res)
            return json.dumps(res) + note
        if tool == "finish":
            log(event="finish", turn=turn); finalize(args.get("report", ""))
            return "submitted; the run is over. Reply with one short closing line and stop."
        return f"unknown tool {tool}"

    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line); mid = req.get("id"); m = req.get("method")
        if m == "initialize":
            res = {"protocolVersion": req["params"].get("protocolVersion", "2024-11-05"), "capabilities": {"tools": {}},
                   "serverInfo": {"name": "tdwa", "version": "1"}}
        elif m == "tools/list":
            res = {"tools": [{"name": t["function"]["name"], "description": t["function"]["description"], "inputSchema": t["function"]["parameters"]} for t in TOOLS]}
        elif m == "tools/call":
            with lock:
                try:
                    text = call(req["params"]["name"], req["params"].get("arguments") or {}); err = False
                except Exception as e:
                    text = f"harness error: {type(e).__name__}: {e}"; err = True; log(event="harness_error", error=text)
            res = {"content": [{"type": "text", "text": text}], "isError": err}
        elif m == "ping":
            res = {}
        elif mid is None:
            continue
        else:
            out.write(json.dumps({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found"}}) + "\n"); out.flush(); continue
        out.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": res}) + "\n"); out.flush()
    if state["sb"] is not None:
        finalize(None)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("--task", required=True); ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default="claude-fable-5-2"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-turns", type=int, default=250); ap.add_argument("--fte-budget", type=int, default=10_000_000)
    ap.add_argument("--val-looks", type=int, default=10); ap.add_argument("--mem-ratio", type=float, default=1.5)
    ap.add_argument("--exec-timeout", type=int, default=1200); ap.add_argument("--token-budget", type=int, default=2_000_000)
    ap.add_argument("--effort", default="high"); ap.add_argument("--dtype", default="float32"); ap.add_argument("--val-max-new-tokens", type=int, default=1024)
    a = ap.parse_args()
    name = f"{a.task}__{a.student.split('/')[-1]}__{a.teacher}__s{a.seed}"
    run_dir = os.path.join(ROOT, "runs", name)
    if os.path.exists(run_dir):
        sys.exit(f"{run_dir} exists; refusing to overwrite")
    os.makedirs(os.path.join(run_dir, "work")); os.makedirs(os.path.join(run_dir, "exams"))
    cfg = {"task": a.task, "student": a.student, "teacher": a.teacher, "seed": a.seed, "fte_budget": a.fte_budget, "mem_ratio": a.mem_ratio, "dtype": a.dtype,
           "token_budget": a.token_budget, "max_turns": a.max_turns, "val_looks": a.val_looks, "gpu": a.gpu, "exec_timeout": a.exec_timeout,
           "scaffold": "claude -p (headless Claude Code), tools via MCP only", "val_max_new_tokens": a.val_max_new_tokens, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    n_dev = sum(1 for _ in open(os.path.join(ROOT, "data", a.task, "dev.jsonl")))
    system = SYSTEM.format(student=a.student, task_text=TASK_TEXT[a.task], n_dev=n_dev, val_looks=a.val_looks, mem_ratio=a.mem_ratio,
                           fte_budget=a.fte_budget, token_budget=a.token_budget, max_turns=a.max_turns)
    system = system.replace("call the `run_python` tool", "call the `run_python` tool (it appears as mcp__tdwa__run_python; the other two tools are mcp__tdwa__official_validation and mcp__tdwa__finish)")
    open(os.path.join(run_dir, "teacher_prompt.txt"), "w").write(system)
    cwd = tempfile.mkdtemp(prefix="tdwa_teacher_")       # empty directory: no project instructions, no files
    mcp = {"mcpServers": {"tdwa": {"command": sys.executable, "args": ["-m", "tdwa.mcp_teach", "serve", run_dir], "cwd": ROOT,
                                   "env": {"PYTHONPATH": ROOT}}}}
    mcp_path = os.path.join(cwd, "mcp.json"); json.dump(mcp, open(mcp_path, "w"))
    tools = "mcp__tdwa__run_python,mcp__tdwa__official_validation,mcp__tdwa__finish"
    cmd = ["claude", "-p", "Begin teaching. Start by calling S.info() through run_python.", "--model", a.teacher, "--system-prompt", system,
           "--tools", "", "--mcp-config", mcp_path, "--strict-mcp-config", "--allowedTools", tools, "--setting-sources", "",
           "--output-format", "stream-json", "--verbose", "--effort", a.effort, "--no-session-persistence"]
    env = dict(os.environ, MCP_TOOL_TIMEOUT="7500000", MCP_TIMEOUT="120000", MAX_MCP_OUTPUT_TOKENS="20000")
    usage = None
    with open(os.path.join(run_dir, "claude_stream.jsonl"), "w") as f:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=open(os.path.join(run_dir, "claude.stderr"), "w"), text=True)
        for line in p.stdout:
            f.write(line); f.flush()
            try:
                e = json.loads(line)
                if e.get("type") == "result":
                    usage = {k: e.get(k) for k in ("usage", "total_cost_usd", "num_turns", "duration_ms", "subtype", "modelUsage")}
            except Exception:
                pass
        p.wait()
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    cfg["teacher_usage"] = usage; cfg["claude_exit_code"] = p.returncode
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    print(json.dumps(cfg, indent=1, default=str)[:3000])


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "serve":
        serve(sys.argv[2])
    else:
        run()
