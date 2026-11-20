"""Teaching run driven by the headless Codex CLI (`codex exec`), with the same three MCP tools as tdwa.mcp_teach.

Driver:  python -m tdwa.codex_teach --task t1 --student google/gemma-3-4b-it --model gpt-6.2-astra --seed 0 --gpu 0
Isolation (Codex has built-in shell/file tools that cannot all be switched off from the command line):
  * codex runs inside a user+mount namespace in which the whole project tree (/mnt/data/users/.../proj), ~/.cache,
    ~/.claude, ~/.config, ~/.ssh, ~/.cli-proxy-api and the user's ~/.codex are replaced by empty directories, so neither
    the model nor its shell can read test/validation items, other runs or dataset caches;
  * a throw-away CODEX_HOME holds only auth.json (no global config, instructions, skills, memories or other MCP servers);
  * codex's own shell sandbox is read-only and web search is disabled;
  * the three teaching tools are served OUTSIDE the namespace by `python -m tdwa.mcp_teach serve <run_dir>` and reach
    codex through a localhost TCP bridge (codex launches a tiny stdio<->TCP relay as its MCP server).
Run name: <task>__<student>__codex-<model>__s<seed>, so these runs never collide with the API-loop Astra runs.
"""
import argparse, json, os, shutil, socket, subprocess, sys, tempfile, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tdwa.teach import SYSTEM, TASK_TEXT  # noqa: E402

HOME = os.path.expanduser("~")
HIDE = [os.path.dirname(os.path.dirname(ROOT))] + [os.path.join(HOME, d) for d in (".cache", ".claude", ".codex", ".config", ".ssh", ".cli-proxy-api")]
RELAY = r'''import socket, sys, threading
s = socket.create_connection(("127.0.0.1", int(sys.argv[1])))
def up():
    while True:
        b = sys.stdin.buffer.readline()
        if not b: break
        s.sendall(b)
    s.shutdown(socket.SHUT_WR)
threading.Thread(target=up, daemon=True).start()
f = s.makefile("rb")
for line in f:
    sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
'''


def serve_tcp(run_dir, port_box, done):
    """Accept one connection from the relay and pipe it to the real MCP server (outside the namespace)."""
    ls = socket.socket(); ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); ls.bind(("127.0.0.1", 0)); ls.listen(1)
    port_box.append(ls.getsockname()[1])
    conn, _ = ls.accept(); ls.close()
    srv = subprocess.Popen([sys.executable, "-m", "tdwa.mcp_teach", "serve", run_dir], cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT),
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(run_dir, "mcp_server.stderr"), "a"))
    def down():
        for line in srv.stdout:
            try:
                conn.sendall(line)
            except OSError:
                break
    threading.Thread(target=down, daemon=True).start()
    f = conn.makefile("rb")
    for line in f:
        srv.stdin.write(line); srv.stdin.flush()
    try:
        srv.stdin.close(); srv.wait(timeout=600)
    except Exception:
        srv.kill()
    done.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--student", required=True); ap.add_argument("--model", default="gpt-6.2-astra")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-turns", type=int, default=250); ap.add_argument("--fte-budget", type=int, default=10_000_000)
    ap.add_argument("--val-looks", type=int, default=10); ap.add_argument("--mem-ratio", type=float, default=1.5)
    ap.add_argument("--exec-timeout", type=int, default=1200); ap.add_argument("--token-budget", type=int, default=2_000_000)
    ap.add_argument("--effort", default="high"); ap.add_argument("--val-max-new-tokens", type=int, default=1024)
    ap.add_argument("--tag", default="", help="suffix for the run name, e.g. e-low (effort sweep)")
    ap.add_argument("--web", action="store_true", help="enable Codex web search (web-search condition only)")
    a = ap.parse_args()
    teacher = f"codex-{a.model}"
    name = f"{a.task}__{a.student.split('/')[-1]}__{teacher}{('_' + a.tag) if a.tag else ''}__s{a.seed}"
    run_dir = os.path.join(ROOT, "runs", name)
    if os.path.exists(run_dir):
        sys.exit(f"{run_dir} exists; refusing to overwrite")
    os.makedirs(os.path.join(run_dir, "work")); os.makedirs(os.path.join(run_dir, "exams"))
    cfg = {"task": a.task, "student": a.student, "teacher": teacher, "teacher_model": a.model, "seed": a.seed, "fte_budget": a.fte_budget,
           "mem_ratio": a.mem_ratio, "token_budget": a.token_budget, "max_turns": a.max_turns, "val_looks": a.val_looks, "gpu": a.gpu,
           "exec_timeout": a.exec_timeout, "val_max_new_tokens": a.val_max_new_tokens, "effort": a.effort, "host": socket.gethostname(),
           "scaffold": "codex exec (headless Codex CLI) in a mount namespace hiding the project; teaching tools via MCP only (TCP relay); "
                       "codex shell sandbox read-only, web search " + ("on" if a.web else "off"), "web": a.web, "tag": a.tag, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    n_dev = sum(1 for _ in open(os.path.join(ROOT, "data", a.task, "dev.jsonl")))
    system = SYSTEM.format(student=a.student, task_text=TASK_TEXT[a.task], n_dev=n_dev, val_looks=a.val_looks, mem_ratio=a.mem_ratio,
                           fte_budget=a.fte_budget, token_budget=a.token_budget, max_turns=a.max_turns)
    system = system.replace("call the `run_python` tool", "call the `run_python` tool of the `tdwa` MCP server (the other two tools are `official_validation` and `finish`)")
    prompt = (system + "\n\nYou have no other tools: do not use shell commands or read local files; everything happens through the tdwa tools. "
              "Work autonomously until you call `finish`; nobody will answer questions. Begin teaching. Start by calling S.info() through run_python.")
    open(os.path.join(run_dir, "teacher_prompt.txt"), "w").write(prompt)

    work = tempfile.mkdtemp(prefix="tdwa_codex_")                    # outside the hidden tree (TMPDIR is on the data disk, not under proj)
    chome = os.path.join(work, "codex_home"); os.makedirs(chome); empty = os.path.join(work, "empty"); os.makedirs(empty)
    cwd = os.path.join(work, "cwd"); os.makedirs(cwd)
    shutil.copy(os.path.join(HOME, ".codex", "auth.json"), os.path.join(chome, "auth.json"))
    open(os.path.join(chome, "config.toml"), "w").write("")
    relay = os.path.join(work, "relay.py"); open(relay, "w").write(RELAY)

    port_box, done = [], threading.Event()
    threading.Thread(target=serve_tcp, args=(run_dir, port_box, done), daemon=True).start()
    while not port_box:
        time.sleep(0.05)
    port = port_box[0]
    codex = ["codex", "exec", "-m", a.model, "--skip-git-repo-check", "--sandbox", "read-only", "--json", "-C", cwd,
             "-c", f"model_reasoning_effort={a.effort}", "-c", "tools.web_search=" + ("true" if a.web else "false"), "-c", "web_search=" + ('"live"' if a.web else '"disabled"'),
             "-c", 'mcp_servers.tdwa.command="/usr/bin/python3"', "-c", f'mcp_servers.tdwa.args=["{relay}","{port}"]',
             "-c", "mcp_servers.tdwa.startup_timeout_sec=900", "-c", 'mcp_servers.tdwa.default_tools_approval_mode="approve"', "-c", "mcp_servers.tdwa.tool_timeout_sec=7500", prompt]
    hide = [h for h in HIDE if os.path.isdir(h)]
    mounts = " && ".join(f"mount --bind {empty} {h}" for h in hide)
    cfg["hidden_from_teacher"] = hide; json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    env = {k: v for k, v in os.environ.items() if not any(s in k for s in ("HF_", "CLIPROXY", "OPENAI_API", "ANTHROPIC", "TDWA"))}
    env.update(CODEX_HOME=chome, HOME=HOME)
    sh = mounts + " && exec \"$@\""
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_output_tokens": 0, "turns": 0}
    t0 = time.time()
    with open(os.path.join(run_dir, "codex_stream.jsonl"), "w") as f:
        p = subprocess.Popen(["unshare", "-rm", "sh", "-c", sh, "sh"] + codex, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=open(os.path.join(run_dir, "codex.stderr"), "w"), text=True)
        for line in p.stdout:
            f.write(line); f.flush()
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") == "turn.completed" and isinstance(e.get("usage"), dict):
                usage["turns"] += 1
                for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
                    usage[k] += e["usage"].get(k) or 0
        p.wait()
    done.wait(timeout=900)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    cfg["teacher_usage"] = usage; cfg["codex_exit_code"] = p.returncode; cfg["driver_wall_seconds"] = round(time.time() - t0)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    shutil.rmtree(chome, ignore_errors=True)                          # drop the copied credentials
    print(json.dumps(cfg, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
