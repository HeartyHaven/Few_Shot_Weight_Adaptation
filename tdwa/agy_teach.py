"""Teaching run driven by the headless Antigravity CLI (`agy`, print mode), with the same three MCP tools as tdwa.mcp_teach.

Driver:  python -m tdwa.agy_teach --task t1 --student Qwen/Qwen3-1.7B --model gemini-4-argon-high --seed 0 --gpu 0 [--tag X]
Leak test (same namespace and settings, a stub tdwa server that answers every tool call with PROBE-OK):
         python -m tdwa.agy_teach probe --model gemini-3.8-flash-low --prompt "..."
Isolation (agy's read-only file tools need no permission, so permission rules alone are not enough):
  * agy runs inside a user + mount + pid namespace (fresh /proc). /tmp is a fresh tmpfs that holds only agy's private HOME,
    an empty cwd, the MCP relay and a read-only bind of the agy binary; the whole user tree (/mnt/data/users/<user>, which
    contains the project, _secret, runs/, runs_void/, the HF cache and TMPDIR), TDWA_USER_ROOT, the user's home (~/.ssh,
    ~/.gemini with the user's own agy config/history, ~/.codex, ~/.claude, ~/.config, ...), /var/tmp and /dev/shm are
    covered by empty read-only tmpfs mounts. The network is kept (the model API needs it).
  * agy's HOME is a private copy containing only the OAuth token file (~/.gemini/antigravity-cli/antigravity-oauth-token),
    a settings.json with this run's permission rules and a ~/.gemini/config/mcp_config.json that names only the tdwa
    server. The user's global agy config is never read or written (no `agy mcp add`). The staged copy is deleted at the end.
  * permissions: allow mcp(tdwa/*); deny command(*), unsandboxed(*), read_file(*), write_file(*), read_url(*),
    execute_url(*), download_url(*); headless mode soft-denies anything else that needs approval.
  * the three teaching tools are served OUTSIDE the namespace by `python -m tdwa.mcp_teach serve <run_dir>` and reach agy
    through a localhost TCP bridge (agy launches a tiny stdio<->TCP relay as its MCP server), as in tdwa.codex_teach.
  * the prompt goes in through stdin as one stream-json user event (a long argv prompt fails with E2BIG). If agy ends a
    turn without calling the tdwa finish tool, the driver sends up to --nudges short "continue" messages.
Run name: <task>__<student>__agy-<model>[_<tag>]__s<seed>.
"""
import argparse, json, os, shutil, socket, subprocess, sys, tempfile, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tdwa.teach import SYSTEM, TASK_TEXT, TOOLS  # noqa: E402

HOME = os.path.expanduser("~")
AGY = shutil.which("agy") or os.path.join(HOME, ".local", "bin", "agy")
TOKEN = os.path.join(HOME, ".gemini", "antigravity-cli", "antigravity-oauth-token")
USER_TREE = os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))      # /mnt/data/users/<user> (holds proj/, TMPDIR, caches)
HIDE = [USER_TREE, os.path.dirname(os.path.dirname(ROOT)), ROOT, os.path.join(ROOT, "_secret"), os.path.join(ROOT, "runs"),
        os.path.join(ROOT, "runs_void"), os.environ.get("TDWA_USER_ROOT", ""), os.environ.get("TMPDIR", ""), HOME, "/var/tmp"]
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
PERMISSIONS = {"allow": [f"mcp(tdwa/{n})" for n in TOOL_NAMES],
               "deny": ["command(*)", "unsandboxed(*)", "read_file(*)", "write_file(*)", "read_url(*)", "execute_url(*)", "download_url(*)"]}
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
# PreToolUse hook (second gate, independent of the permission rules): only calls of the tdwa MCP tools may run; every
# built-in tool (file viewers, grep/find, shell, browser, web search, URL reading, subagents, ...) is refused.
GATE = r'''import json, sys
mode = sys.argv[1] if len(sys.argv) > 1 else "strict"
try:
    d = json.load(sys.stdin)
except Exception:
    d = {}
tc = d.get("toolCall") or {}
n = str(tc.get("name") or "")
args = json.dumps(tc.get("args") or {})
ok = n.startswith("mcp_tdwa_") or (n in ("call_mcp_tool", "mcp_tool") and '"tdwa"' in args)
if mode == "debug":
    ok = False
reason = ("tool %r is disabled in this experiment; only the tdwa MCP tools (run_python, official_validation, finish) are available" % n)
if mode == "debug":
    reason += " [gate-debug args=%s]" % args[:300]
print(json.dumps({"decision": "allow" if ok else "deny", "reason": reason}))
'''
# Runs as "root" of a fresh user namespace (unshare -rm --pid --fork --mount-proc): read the staged files into memory, build
# the private /tmp, hide everything else, then exec agy. Errors abort before agy starts.
NS_INIT = r'''import json, os, subprocess, sys
spec = json.load(open(sys.argv[1]))
files = {rel: open(src, "rb").read() for rel, src in spec["files"].items()}
for src in spec["files"].values():                      # the staged copies (incl. the token) are not needed any more
    os.remove(src)
def mnt(*a):
    subprocess.run(["mount", *a], check=True)
mnt("-t", "tmpfs", "-o", "mode=700,size=4g", "tmpfs", "/tmp")
for d in ("h", "cwd", "bin"):
    os.makedirs("/tmp/" + d, mode=0o700)
for rel, data in files.items():
    p = "/tmp/" + rel; os.makedirs(os.path.dirname(p), mode=0o700, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT, 0o600); os.write(fd, data); os.close(fd)
open("/tmp/bin/agy", "w").close(); os.chmod("/tmp/bin/agy", 0o755)
mnt("--bind", spec["agy"], "/tmp/bin/agy")
subprocess.run(["mount", "-o", "remount,bind,ro", "/tmp/bin/agy"])
hidden = []
for h in spec["hide"]:
    if h and os.path.isdir(h) and not os.path.realpath(h).startswith("/tmp"):
        mnt("-t", "tmpfs", "-o", "ro,size=4k,mode=555", "tmpfs", h); hidden.append(h)
if os.path.isdir("/dev/shm"):
    mnt("-t", "tmpfs", "-o", "mode=700,size=64m", "tmpfs", "/dev/shm")
sys.stderr.write("[ns_init] hidden: " + json.dumps(hidden) + "\n"); sys.stderr.flush()
os.chdir("/tmp/cwd")
os.execve("/tmp/bin/agy", ["agy", *spec["args"]], spec["env"])
'''


def stub_serve():
    """MCP server with the real tool list whose calls only echo (leak test; no GPU, no student)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line); mid = req.get("id"); m = req.get("method")
        if m == "initialize":
            res = {"protocolVersion": req["params"].get("protocolVersion", "2024-11-05"), "capabilities": {"tools": {}}, "serverInfo": {"name": "tdwa", "version": "1"}}
        elif m == "tools/list":
            res = {"tools": [{"name": t["function"]["name"], "description": t["function"]["description"], "inputSchema": t["function"]["parameters"]} for t in TOOLS]}
        elif m == "tools/call":
            res = {"content": [{"type": "text", "text": "PROBE-OK " + req["params"]["name"] + " " + json.dumps(req["params"].get("arguments") or {})[:200]}], "isError": False}
        elif m == "ping":
            res = {}
        elif mid is None:
            continue
        else:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found"}}) + "\n"); sys.stdout.flush(); continue
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": res}) + "\n"); sys.stdout.flush()


def serve_tcp(server_cmd, stderr_path, port_box, done):
    """Accept one connection from the relay and pipe it to the real MCP server (outside the namespace)."""
    ls = socket.socket(); ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); ls.bind(("127.0.0.1", 0)); ls.listen(1)
    port_box.append(ls.getsockname()[1])
    conn, _ = ls.accept(); ls.close(); port_box.append("connected")
    srv = subprocess.Popen(server_cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(stderr_path, "a"))
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
        srv.stdin.close(); srv.wait(timeout=1800)
    except Exception:
        srv.kill()
    done.set()


def run_agy(model, prompt, server_cmd, log_dir, effort=None, finished=lambda: True, nudges=0, nudge_text="", gate_mode="strict"):
    """Run agy (print mode, stream-json in/out) in the isolating namespace with tdwa as its only MCP server.
    Returns (usage, exit_code, info). Writes agy_stream.jsonl, agy.stderr, mcp_server.stderr into log_dir."""
    if not os.path.isfile(TOKEN):
        sys.exit("no agy login found (" + TOKEN + "); run `agy` once interactively to sign in")
    port_box, done = [], threading.Event()
    threading.Thread(target=serve_tcp, args=(server_cmd, os.path.join(log_dir, "mcp_server.stderr"), port_box, done), daemon=True).start()
    while not port_box:
        time.sleep(0.05)
    stage = tempfile.mkdtemp(prefix="tdwa_agy_")                         # mode 700; hidden inside the namespace; removed below
    try:
        settings = {"permissions": PERMISSIONS, "toolPermission": "request-review", "allowNonWorkspaceAccess": False}
        mcp = {"mcpServers": {"tdwa": {"command": "/usr/bin/python3", "args": ["-I", "/tmp/relay.py", str(port_box[0])], "timeoutSeconds": 7500,
                                       "tools": {n: {"eager": True} for n in TOOL_NAMES}}}}
        for fn, obj in (("settings.json", settings), ("mcp_config.json", mcp)):
            json.dump(obj, open(os.path.join(stage, fn), "w"), indent=1)
        open(os.path.join(stage, "relay.py"), "w").write(RELAY); open(os.path.join(stage, "gate.py"), "w").write(GATE)
        hooks = {"tdwa-gate": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": "/usr/bin/python3 -I /tmp/gate.py " + gate_mode, "timeout": 20}]}]}}
        json.dump(hooks, open(os.path.join(stage, "hooks.json"), "w"), indent=1)
        shutil.copy(TOKEN, os.path.join(stage, "token")); os.chmod(os.path.join(stage, "token"), 0o600)
        args = ["--model", model, "--input-format", "stream-json", "--output-format", "stream-json", "--print=", "--settings", "/tmp/h/.gemini/antigravity-cli/settings.json",
                "--disable-slash-commands"] + (["--effort", effort] if effort else [])
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp/h", "TMPDIR": "/tmp", "LANG": os.environ.get("LANG", "C.UTF-8"), "TERM": "dumb",
               "USER": os.environ.get("USER", "user"), "AGY_CLI_HIDE_ACCOUNT_INFO": "1"}
        spec = {"agy": os.path.realpath(AGY), "args": args, "env": env, "hide": [h for h in HIDE if h],
                "files": {"h/.gemini/antigravity-cli/antigravity-oauth-token": os.path.join(stage, "token"),
                          "h/.gemini/antigravity-cli/settings.json": os.path.join(stage, "settings.json"),
                          "h/.gemini/config/mcp_config.json": os.path.join(stage, "mcp_config.json"), "h/.gemini/config/hooks.json": os.path.join(stage, "hooks.json"),
                          "relay.py": os.path.join(stage, "relay.py"), "gate.py": os.path.join(stage, "gate.py")}}
        json.dump(spec, open(os.path.join(stage, "spec.json"), "w"))
        info = {"settings": settings, "hooks": hooks, "mcp_config": {"tdwa": {k: v for k, v in mcp["mcpServers"]["tdwa"].items()}}, "agy_args": args,
                "hidden_from_teacher": spec["hide"] + ["/tmp (fresh tmpfs)", "/dev/shm", "/proc (own pid namespace)"]}
        init = os.path.join(stage, "ns_init.py"); open(init, "w").write(NS_INIT)
        usage = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "cache_read_tokens": 0, "total_tokens": 0, "turns": 0, "num_turns": 0}
        results, sent = [], 0
        with open(os.path.join(log_dir, "agy_stream.jsonl"), "w") as f:
            p = subprocess.Popen(["unshare", "-rm", "--pid", "--fork", "--mount-proc", "/usr/bin/python3", "-I", init, os.path.join(stage, "spec.json")],
                                 cwd="/", env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(log_dir, "agy.stderr"), "w"), text=True)
            def send(text):
                p.stdin.write(json.dumps({"event": "user", "message": {"content": text}}) + "\n"); p.stdin.flush()
            send(prompt)
            for line in p.stdout:
                f.write(line); f.flush()
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("event") == "init":                               # agy has read everything; drop the staged copy (token) now
                    shutil.rmtree(stage, ignore_errors=True)
                if e.get("event") != "result":
                    continue
                r = e.get("result") or {}
                results.append({k: r.get(k) for k in ("status", "error", "duration_seconds", "num_turns", "usage", "conversation_id")})
                usage["turns"] += 1; usage["num_turns"] += r.get("num_turns") or 0
                for k in ("input_tokens", "output_tokens", "thinking_tokens", "cache_read_tokens", "total_tokens"):
                    usage[k] += (r.get("usage") or {}).get(k) or 0
                if not finished() and sent < nudges and r.get("status") in ("SUCCESS", None):
                    sent += 1; send(nudge_text)
                else:
                    try:
                        p.stdin.close()
                    except OSError:
                        pass
            p.wait()
        usage["nudges_sent"] = sent; info["results"] = results
        if "connected" in port_box:
            done.wait(timeout=1800)
        return usage, p.returncode, info
    finally:
        shutil.rmtree(stage, ignore_errors=True)                          # drop the copied credentials


def probe(argv):
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--model", default="gemini-3.8-flash-low")
    ap.add_argument("--prompt", required=True); ap.add_argument("--gate", default="strict"); ap.add_argument("--out", required=True, help="directory for the probe logs (outside the project is fine)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    usage, rc, info = run_agy(a.model, a.prompt, [sys.executable, "-m", "tdwa.agy_teach", "stubserve"], a.out, gate_mode=a.gate)
    print(json.dumps({"exit_code": rc, "usage": usage, "hidden": info["hidden_from_teacher"]}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--student", required=True); ap.add_argument("--model", default="gemini-4-argon-high")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-turns", type=int, default=250); ap.add_argument("--fte-budget", type=int, default=10_000_000)
    ap.add_argument("--val-looks", type=int, default=10); ap.add_argument("--mem-ratio", type=float, default=1.5)
    ap.add_argument("--exec-timeout", type=int, default=1200); ap.add_argument("--token-budget", type=int, default=2_000_000)
    ap.add_argument("--effort", default=None, help="agy --effort (the model name usually fixes the effort already)")
    ap.add_argument("--dtype", default="float32"); ap.add_argument("--val-max-new-tokens", type=int, default=1024)
    ap.add_argument("--nudges", type=int, default=3, help="max 'continue' messages if agy ends a turn without calling finish")
    ap.add_argument("--tag", default="", help="suffix for the run name, e.g. smoke")
    a = ap.parse_args()
    teacher = f"agy-{a.model}"
    name = f"{a.task}__{a.student.split('/')[-1]}__{teacher}{('_' + a.tag) if a.tag else ''}__s{a.seed}"
    run_dir = os.path.join(ROOT, "runs", name)
    if os.path.exists(run_dir):
        sys.exit(f"{run_dir} exists; refusing to overwrite")
    os.makedirs(os.path.join(run_dir, "work")); os.makedirs(os.path.join(run_dir, "exams"))
    cfg = {"task": a.task, "student": a.student, "teacher": teacher, "teacher_model": a.model, "seed": a.seed, "fte_budget": a.fte_budget,
           "mem_ratio": a.mem_ratio, "dtype": a.dtype, "token_budget": a.token_budget, "max_turns": a.max_turns, "val_looks": a.val_looks, "gpu": a.gpu,
           "exec_timeout": a.exec_timeout, "val_max_new_tokens": a.val_max_new_tokens, "effort": a.effort, "host": socket.gethostname(),
           "scaffold": "agy --print (headless Antigravity CLI, stream-json) in a user+mount+pid namespace hiding the user tree and home; private agy HOME "
                       "with only the OAuth token; teaching tools via MCP only (TCP relay); shell, file, URL and web permissions denied",
           "tag": a.tag, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    n_dev = sum(1 for _ in open(os.path.join(ROOT, "data", a.task, "dev.jsonl")))
    system = SYSTEM.format(student=a.student, task_text=TASK_TEXT[a.task], n_dev=n_dev, val_looks=a.val_looks, mem_ratio=a.mem_ratio,
                           fte_budget=a.fte_budget, token_budget=a.token_budget, max_turns=a.max_turns)
    system = system.replace("call the `run_python` tool", "call the `run_python` tool of the `tdwa` MCP server (the other two tools are `official_validation` and `finish` of the same server)")
    prompt = (system + "\n\nYou have no other tools: your built-in shell, file, browser and web tools are disabled, so do not use them; everything happens "
              "through the three tdwa MCP tools. Your built-in `finish` tool does NOT end the run: only the `finish` tool of the tdwa MCP server does. "
              "Work autonomously until you call the tdwa `finish`; nobody will answer questions. Begin teaching. Start by calling S.info() through run_python.")
    open(os.path.join(run_dir, "teacher_prompt.txt"), "w").write(prompt)
    nudge = ("Continue teaching autonomously with the tdwa MCP tools (run_python, official_validation). The run only ends when you call the "
             "tdwa MCP `finish` tool with your final report; if you are done, call it now.")
    t0 = time.time()
    finished = lambda: os.path.exists(os.path.join(run_dir, "final", "report.md"))
    usage, rc, info = run_agy(a.model, prompt, [sys.executable, "-m", "tdwa.mcp_teach", "serve", run_dir], run_dir, a.effort, finished, a.nudges, nudge)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    cfg.update(teacher_usage=usage, agy_exit_code=rc, agy_results=info["results"], hidden_from_teacher=info["hidden_from_teacher"],
               agy_settings=info["settings"], agy_mcp=info["mcp_config"], agy_args=info["agy_args"], driver_wall_seconds=round(time.time() - t0))
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    print(json.dumps(cfg, indent=1, default=str)[:3000])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stubserve":
        stub_serve()
    elif len(sys.argv) > 1 and sys.argv[1] == "probe":
        probe(sys.argv[1:])
    else:
        main()
