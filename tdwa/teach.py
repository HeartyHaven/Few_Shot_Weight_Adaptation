"""One teaching run: a teacher model (via an OpenAI-compatible endpoint) teaches a student by editing its weights.

Usage: python -m tdwa.teach --task t2 --student google/gemma-3-1b-it --teacher gpt-6-astra --seed 0
Everything is logged under runs/<task>__<student>__<teacher>__s<seed>/ (see DESIGN.md section 10).
"""
import argparse, glob, json, os, shlex, shutil, subprocess, sys, tempfile, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tdwa import grade as G  # noqa: E402

HF = os.environ.get("TDWA_HF", "/mnt/data2/users/lang.gao/huggingface_cache")   # per-host override
VENV = os.environ.get("TDWA_VENV", "/mnt/data2/users/lang.gao/Weight_Programming_Eval/.venv")
USER_ROOT = os.environ.get("TDWA_USER_ROOT", "/mnt/data2/users/lang.gao")

TASK_TEXT = {
    "t2": ("Object-state tracking. Each item lists N people each holding one object, then a sequence of pairwise swaps, and asks what "
           "one person holds at the end (multiple choice). N is 3, 5 or 7 and the number of swaps equals N. The hidden test also "
           "contains longer sequences and N=9, and differently worded items of the same kind."),
    "t1": ("Rule-based clinical scores from MedCalc-Bench. Each item is a patient note plus a question asking for one score. The 8 "
           "calculators you must teach are: CURB-65, SIRS criteria, PERC rule, Wells' criteria for PE, Centor (modified/McIsaac), "
           "Child-Pugh, HAS-BLED and CHA2DS2-VASc. The answer is an integer (Wells: may be x.5). The hidden test covers these 8 and, "
           "as a side-effect check, 11 other calculators you are not asked to teach."),
}

SYSTEM = """You are the TEACHER in a research experiment on teaching a small language model by editing its weights directly.

THE STUDENT is {student} (float32, on one GPU). THE TASK: {task_text}

YOUR GOAL: make the student reliably good at this task when it is asked in the plain official prompt format (S.format_prompt), with NO procedure, hints or examples in the prompt. You do this by changing the student's weights. You are a teacher, not a trainer: you write down how the task should be done, examine the student, work out precisely why it fails (use any interpretability method you can implement: activation inspection, logit lens, activation patching, ablations, comparing runs with and without a procedure in the prompt, attention analysis, SVD of weights...), then design a targeted weight edit, predict its effect, apply it, and test whether your prediction held. An independent evaluator will afterwards test the edited student on a hidden test set, measure side effects on unrelated abilities, and compare your cost against LoRA fine-tuning and distillation.

HOW YOU WORK: call the `run_python` tool. Code runs in a persistent Python session (variables persist) with `S`, `torch`, `np`, `json` defined. Output is truncated to ~8000 characters, so print summaries. There is no network. The working directory is yours for notes and scripts.

StudentAPI `S` (use help(S.method) for details):
  S.info()                         budgets, limits, current state
  S.model, S.tok                   the raw Hugging Face model and tokenizer; S.params() -> dict name -> parameter
  S.dev                            the ONLY real labelled data you get: {n_dev} items (dicts: id, group, question, answer, ...)
  S.format_prompt(item, mode)      the official user message; S.render(user, system=None) applies the chat template
  S.chat(users, max_new_tokens)    greedy generation for raw user messages (any prompt you like, e.g. with a procedure, for analysis)
  S.quiz(items, label, skill)      official-format quiz on dev items or items you write yourself; graded and logged per item
  S.grade(item, output)            grader used everywhere
  S.add_bias(site)                 add a new zero bias vector on a module output, e.g. 'model.layers.13.mlp' (the MLPs have no bias)
  S.regression_check()             perplexity on generic text + a canary quiz, for the current weights
  S.commit_edit(meta)              register the current uncommitted weight changes as ONE edit (see rules)
  S.discard()                      drop uncommitted changes;  S.rollback(to_edit_id, reason)  undo accepted edits
Other tools: `official_validation` runs a hidden validation set (same distribution as the test) and returns only scores; you get {val_looks} looks in total. `finish` ends the run with your final report.

RULES (the run is audited from the logs; violations void the run):
1. No conventional training. Do not run gradient descent or any optimiser over an adapter or over the model on a dataset. Allowed operation classes: analytic (computations on weights), forward_observation, closed_form (least squares, mean differences, projections fitted on activations), zeroth_order_search (only over a small set of parameters you have selected for a stated reason), local_gradient (backprop used only to obtain a target vector at one site, ROME-style), new_bias_parameter.
2. Nothing written into the student may be obtained from a large amount of data. Real labelled data is limited to S.dev. You may write your own items from your own understanding of the task, at most 300 labelled items in total across the run; say which items each edit used.
3. Compute: GPU memory is hard-capped at {mem_ratio}x the student's inference peak (you will get an out-of-memory error beyond it), and you have {fte_budget:,} student forward tokens in total (S.info() shows what is left; official validation and the regression guard do not count).
4. Every weight change you want to keep must go through S.commit_edit(meta) with meta = {{"hypothesis": what is wrong and where, with the evidence; "method": exactly how the new values were computed; "op_class": one of S.op_classes; "prediction": which behaviours should change and which must not, with expected numbers; "data_used": ids of dev items / description of your own items}}. The harness records every changed position with its old and new value, and runs a regression guard; an edit that raises generic perplexity by more than 10% or lowers the canary by more than 3 points is rejected and reverted. After committing, test your prediction with a quiz and record whether it held.
5. Do not try to read files outside your working directory, the validation or test data, or the harness internals.

REQUIRED TEACHING RECORD (files in your working directory, keep them current; they are part of the deliverable):
  syllabus.md        the procedure for the task as you would teach it, the sub-skills it decomposes into, and how each is checked
  curriculum_log.md  dated entries: each diagnostic quiz and what it showed (error categories: E1 does not know the rule / E2 knows but does not execute / E3 a sub-skill is missing / E4 answer-format problem / E5 other), each localisation experiment and its result, each edit proposal with its prediction, the measured outcome, and how you changed the plan as a result. Record failures and reverted edits too.
Suggested loop: syllabus -> diagnostic exam per sub-skill (also try the student WITH your procedure in the prompt, to see what a prompt can and cannot fix) -> localise the main failure -> propose an edit with a prediction -> apply, quiz, regression check, commit or discard -> official validation after each teaching unit -> adjust the plan. Stop when validation stops improving, when you run low on budget, or when you judge the student has learned the task. Negative results are acceptable and valuable if they are well diagnosed: if something cannot be taught by editing, show the evidence for which sub-skill is the obstacle.

When done, call `finish` with a final report (markdown): what you taught, the student's level over time, a one-line-per-edit table, error analysis before and after, failed attempts, what remains unlearned, and total cost.
You have a budget of {token_budget:,} teacher tokens and {max_turns} tool calls. Be economical with output you print."""

TOOLS = [
    {"type": "function", "function": {"name": "run_python", "description": "Execute Python in the persistent teaching sandbox (S, torch, np, json available).",
                                       "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "official_validation", "description": "Run the hidden validation set on the current committed weights. Returns scores only. Limited number of looks.",
                                       "parameters": {"type": "object", "properties": {"note": {"type": "string", "description": "why you are checking now"}}, "required": ["note"]}}},
    {"type": "function", "function": {"name": "finish", "description": "End the teaching run and submit the final report (markdown).",
                                       "parameters": {"type": "object", "properties": {"report": {"type": "string"}}, "required": ["report"]}}},
]


class Sandbox:
    """Kernel subprocess in a user+mount+network namespace. Only the run dir, the student weights, the code and the
    public data are mounted; the rest of the user's tree (secrets, other runs, dataset caches) is hidden."""

    def __init__(self, cfg, run_dir, model_dir, gpu, timeout):
        self.timeout = timeout; self.run_dir = run_dir
        self.mnt = tempfile.mkdtemp(prefix="tdwa_sb_")
        binds = {"run": run_dir, "model": model_dir, "code/tdwa": os.path.join(ROOT, "tdwa"), "data": os.path.join(ROOT, "data"), "venv": VENV,
                 "blobs": os.path.join(HF, "blobs")}   # large model files are links into the shared content-addressed blob store
        for k in binds:
            os.makedirs(os.path.join(self.mnt, k), exist_ok=True)
        empty = os.path.join(self.mnt, ".empty"); os.makedirs(empty)
        snap = glob.glob(os.path.join(model_dir, "snapshots", "*"))[0]
        inner = dict(cfg, run_dir=f"{self.mnt}/run", model_dir=f"{self.mnt}/model/snapshots/{os.path.basename(snap)}",
                     code_root=f"{self.mnt}/code", data_dir=f"{self.mnt}/data")
        json.dump(inner, open(os.path.join(run_dir, "work", ".cfg.json"), "w"))
        home = os.path.expanduser("~")
        hide = [USER_ROOT] + [os.path.join(home, d) for d in (".cache", ".claude", ".codex", ".config", ".ssh", "cliproxyapi")]
        hide = [h for h in hide if os.path.isdir(h)]
        mounts = [f"mount --bind {shlex.quote(v)} {shlex.quote(os.path.join(self.mnt, k))}" for k, v in binds.items()]
        mounts += [f"mount --bind {shlex.quote(empty)} {shlex.quote(h)}" for h in hide]
        self.hidden = hide
        cmd = " && ".join(mounts) + f" && cd {self.mnt}/run/work && exec {self.mnt}/venv/bin/python -I {self.mnt}/code/tdwa/kernel.py"
        env = {k: v for k, v in os.environ.items() if not any(s in k for s in ("HF_", "CLIPROXY", "OPENAI", "ANTHROPIC", "TOKEN", "KEY"))}
        env.update(TDWA_CFG=f"{self.mnt}/run/work/.cfg.json", CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
        self._launch = lambda: subprocess.Popen(["unshare", "-rmn", "sh", "-c", cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                stderr=open(os.path.join(run_dir, "kernel.stderr"), "a"), text=True, env=env)
        self.restarts = 0; self.max_restarts = 8
        self.proc = self._launch()
        self.ready = self._read(900)

    def _restart(self):
        """Start a fresh kernel after a crash/timeout. The kernel restores accepted edits and cost counters from disk."""
        if self.restarts >= self.max_restarts:
            return False
        self.restarts += 1
        try:
            self.proc.kill(); self.proc.wait(timeout=30)
        except Exception:
            pass
        self.proc = self._launch()
        return bool(self._read(900).get("ready"))

    def _read(self, timeout):
        res = {}
        t = threading.Thread(target=lambda: res.__setitem__("line", self.proc.stdout.readline()), daemon=True)
        t.start(); t.join(timeout)
        if t.is_alive():
            self.proc.kill()
            return {"ok": False, "dead": True, "stdout": "", "stderr": f"call exceeded {timeout}s; the sandbox was killed and its state is lost"}
        if not res.get("line"):
            return {"ok": False, "dead": True, "stdout": "", "stderr": "sandbox died (see kernel.stderr)"}
        return json.loads(res["line"])

    def _rpc(self, obj, timeout=None):
        try:
            self.proc.stdin.write(json.dumps(obj) + "\n"); self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return {"ok": False, "dead": True, "stdout": "", "stderr": "sandbox is not running"}
        return self._read(timeout or self.timeout)

    def rpc(self, obj, timeout=None):
        r = self._rpc(obj, timeout)
        if r.get("dead"):
            if self._restart():
                if obj.get("cmd") == "finish":
                    return self._rpc(obj, timeout)
                tail = ""
                try:
                    tail = open(os.path.join(self.run_dir, "kernel.stderr")).read()[-1200:]
                except Exception:
                    pass
                msg = (r["stderr"] + "\nThe sandbox process crashed and has been RESTARTED. Accepted edits and the cost counters were restored; "
                       "uncommitted weight changes and all your Python variables are gone (files in the working directory are kept). "
                       "A CUDA device-side assert (e.g. an out-of-range index on the GPU) kills the process, so check indices first.\n"
                       "Last lines of the crash log:\n" + tail)
                r = {"ok": False, "restarted": True, "stdout": "", "stderr": msg, "error": msg}
            else:
                r["stderr"] += " (restart limit reached or restart failed)"
        return r

    def close(self):
        try:
            self.proc.stdin.close(); self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()
        shutil.rmtree(self.mnt, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--student", required=True); ap.add_argument("--teacher", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gpu", default="0")
    ap.add_argument("--token-budget", type=int, default=2_000_000); ap.add_argument("--max-turns", type=int, default=250)
    ap.add_argument("--fte-budget", type=int, default=10_000_000); ap.add_argument("--val-looks", type=int, default=10)
    ap.add_argument("--mem-ratio", type=float, default=1.5); ap.add_argument("--exec-timeout", type=int, default=1200)
    ap.add_argument("--effort", default=None); ap.add_argument("--val-max-new-tokens", type=int, default=1024); ap.add_argument("--scripted", default=None, help="path to a .py file list for a fake teacher (harness test)")
    a = ap.parse_args()

    name = f"{a.task}__{a.student.split('/')[-1]}__{a.teacher}__s{a.seed}"
    run_dir = os.path.join(ROOT, "runs", name)
    if os.path.exists(run_dir):
        sys.exit(f"{run_dir} exists; refusing to overwrite")
    os.makedirs(os.path.join(run_dir, "work")); os.makedirs(os.path.join(run_dir, "exams"))
    secret_dir = os.path.join(ROOT, "_secret", "exams", name); os.makedirs(secret_dir, exist_ok=True)
    cfg = {"task": a.task, "student": a.student, "teacher": a.teacher, "seed": a.seed, "fte_budget": a.fte_budget, "mem_ratio": a.mem_ratio,
           "token_budget": a.token_budget, "max_turns": a.max_turns, "val_looks": a.val_looks, "val_max_new_tokens": a.val_max_new_tokens, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    tlog = open(os.path.join(run_dir, "transcript.jsonl"), "a")

    def log(**kw):
        tlog.write(json.dumps({"t": time.time(), **kw}, ensure_ascii=False, default=str) + "\n"); tlog.flush()

    model_dir = os.path.join(HF, "models--" + a.student.replace("/", "--"))
    sb = Sandbox(cfg, run_dir, model_dir, a.gpu, a.exec_timeout)
    log(event="sandbox_ready", ready=sb.ready, hidden=sb.hidden)
    if not sb.ready.get("ready"):
        sys.exit("sandbox failed to start: " + json.dumps(sb.ready)[:500])
    info = sb.ready["info"]
    val = [json.loads(l) for l in open(os.path.join(ROOT, "_secret", a.task, "val.jsonl"))]
    looks = [0]

    def official_validation(note):
        if looks[0] >= a.val_looks:
            return {"error": "no validation looks left"}
        r = sb.rpc({"cmd": "generate", "prompts": [G.format_prompt(it) for it in val], "max_new_tokens": cfg.get("val_max_new_tokens", 2048)}, timeout=7200)
        if not r.get("ok"):
            return {"error": r.get("error") or r.get("stderr")}
        looks[0] += 1
        rows = [{"id": it["id"], "group": it["group"], "answer": it["answer"], "output": o, **G.grade(it, o)} for it, o in zip(val, r["outputs"])]
        by = {}
        for x in rows:
            by.setdefault(x["group"], []).append(x["correct"])
        res = {"look": looks[0], "looks_left": a.val_looks - looks[0], "after_edit": r["after_edit"], "n": len(rows),
               "accuracy": round(sum(x["correct"] for x in rows) / len(rows), 4),
               "by_group": {k: round(sum(v) / len(v), 4) for k, v in sorted(by.items())},
               "no_final_answer_rate": round(sum(x["pred"] is None for x in rows) / len(rows), 4)}
        json.dump({**res, "note": note}, open(os.path.join(run_dir, "exams", f"{looks[0]:02d}.json"), "w"), indent=1)
        with open(os.path.join(secret_dir, f"{looks[0]:02d}.jsonl"), "w") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        return res

    system = SYSTEM.format(student=a.student, task_text=TASK_TEXT[a.task], n_dev=info["n_dev"], val_looks=a.val_looks, mem_ratio=a.mem_ratio,
                           fte_budget=a.fte_budget, token_budget=a.token_budget, max_turns=a.max_turns)
    open(os.path.join(run_dir, "teacher_prompt.txt"), "w").write(system)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0, "api_calls": 0, "budget_tokens": 0}
    report = None; turns = 0; t0 = time.time()

    def run_code(code):
        r = sb.rpc({"code": code, "call": turns})
        out = (r.get("stdout") or "") + (("\n[stderr]\n" + r["stderr"]) if r.get("stderr") else "")
        return (out.strip() or "(no output)") + f"\n[{json.dumps(r.get('stat', {}))}]", r

    if a.scripted:      # harness self-test without any teacher model
        for i, path in enumerate(a.scripted.split(",")):
            if path == "VALIDATE":
                print(official_validation("scripted")); continue
            out, r = run_code(open(path).read()); turns += 1
            log(event="tool", name="run_python", code=open(path).read(), result=r); print(f"--- {path}\n{out}")
    else:
        from openai import OpenAI
        client = OpenAI(base_url=os.environ["CLIPROXY_BASE_URL"], api_key=os.environ["CLIPROXY_API_KEY"], timeout=1800)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": "Begin. Initial state:\n" + json.dumps(info, indent=1)}]
        last_ctx = 0; warned = False; skip_fresh = False
        while turns < a.max_turns:
            kw = {"model": a.teacher, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "max_tokens": 32000}
            if a.effort:
                kw["reasoning_effort"] = a.effort
            for attempt in range(6):
                try:
                    resp = client.chat.completions.create(**kw); break
                except Exception as e:
                    log(event="api_error", attempt=attempt, error=str(e)[:800]); time.sleep(20 * (attempt + 1))
            else:
                log(event="abort", reason="api_errors"); break
            u = resp.usage; pt, ct = u.prompt_tokens or 0, u.completion_tokens or 0
            usage["prompt_tokens"] += pt; usage["completion_tokens"] += ct; usage["api_calls"] += 1
            try:
                usage["cached_tokens"] += u.prompt_tokens_details.cached_tokens or 0
                usage["reasoning_tokens"] += u.completion_tokens_details.reasoning_tokens or 0
            except Exception:
                pass
            # budget counts each token once: newly added context + everything the teacher writes
            usage["budget_tokens"] += ct + (0 if skip_fresh else max(0, pt - last_ctx))
            skip_fresh = False; last_ctx = pt + ct
            msg = resp.choices[0].message
            tcs = [tc.model_dump() for tc in (msg.tool_calls or [])]
            messages.append({"role": "assistant", "content": msg.content or "", **({"tool_calls": tcs} if tcs else {})})
            log(event="assistant", turn=turns, content=msg.content, tool_calls=tcs, usage={"prompt": pt, "completion": ct}, cumulative=dict(usage))
            if not tcs:
                messages.append({"role": "user", "content": "Continue teaching. When you are done, call the `finish` tool with your final report."})
                turns += 1; continue
            done = False
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {"code": tc.function.arguments}
                if tc.function.name == "run_python":
                    out, r = run_code(args.get("code", ""))
                    log(event="tool", turn=turns, name="run_python", code=args.get("code", ""), result=r)
                    done = done or bool(r.get("dead"))
                elif tc.function.name == "official_validation":
                    res = official_validation(args.get("note", "")); out = json.dumps(res)
                    log(event="tool", turn=turns, name="official_validation", note=args.get("note"), result=res)
                elif tc.function.name == "finish":
                    report = args.get("report", ""); out = "submitted"; done = True
                    log(event="finish", turn=turns)
                else:
                    out = f"unknown tool {tc.function.name}"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            turns += 1
            if done:
                break
            left_tok = a.token_budget - usage["budget_tokens"]; left_turns = a.max_turns - turns
            if not warned and (left_tok < 0.12 * a.token_budget or left_turns <= 15):
                warned = True
                messages.append({"role": "user", "content": f"Budget notice: about {max(0, left_tok):,} teacher tokens and {left_turns} tool calls remain. Bring the teaching records up to date, commit or discard pending changes, and call `finish` with your final report."})
            if left_tok <= 0:
                log(event="abort", reason="token_budget"); break
            if pt > 160_000:    # keep the context bounded: shorten old tool outputs (full text stays in transcript.jsonl)
                for m in messages[2:-40]:
                    if m["role"] == "tool" and len(m["content"]) > 400:
                        m["content"] = m["content"][:300] + "\n[older output shortened by the harness; keep important findings in curriculum_log.md]"
                skip_fresh = True; log(event="context_shortened", prompt_tokens=pt)

    fin = sb.rpc({"cmd": "finish"}, timeout=1800); sb.close()
    os.makedirs(os.path.join(run_dir, "final"), exist_ok=True)
    open(os.path.join(run_dir, "final", "report.md"), "w").write(report or "(the teacher did not submit a final report)")
    cfg.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), wall_seconds=round(time.time() - t0), turns=turns, teacher_usage=usage,
               validation_looks=looks[0], finish=fin, report_submitted=report is not None)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    log(event="end", **cfg)
    print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
