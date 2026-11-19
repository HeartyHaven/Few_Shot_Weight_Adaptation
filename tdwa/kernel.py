"""Sandbox kernel: holds the student model and exposes the StudentAPI object `S` to teacher code.

Runs inside an isolated namespace (no network; secrets, other runs and dataset caches are not mounted).
Protocol: JSON lines on stdin/stdout.
  {"code": str}                       -> exec teacher code in the persistent namespace
  {"cmd": "generate", "prompts": [...], "max_new_tokens": n}  -> parent-driven generation (official validation)
  {"cmd": "finish"}                   -> discard uncommitted changes, write the final cumulative edit, exit
"""
import contextlib, io, json, math, os, sys, time, traceback

_OUT = os.fdopen(os.dup(1), "w")   # protocol channel; everything else that prints goes to stderr
os.dup2(2, 1)

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

CFG = json.load(open(os.environ["TDWA_CFG"]))
sys.path.insert(0, CFG["code_root"])
from tdwa import grade as G  # noqa: E402

RUN = CFG["run_dir"]; WORK = os.path.join(RUN, "work")
for d in ("edits", "quizzes"):
    os.makedirs(os.path.join(RUN, d), exist_ok=True)
OP_CLASSES = ["analytic", "forward_observation", "closed_form", "zeroth_order_search", "local_gradient", "new_bias_parameter"]
REQUIRED_META = ["hypothesis", "method", "op_class", "prediction", "data_used"]

tok = AutoTokenizer.from_pretrained(CFG["model_dir"])
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
DTYPE = getattr(torch, CFG.get("dtype", "float32"))       # float32 for students up to 4B; bfloat16 for 8B+ (memory)
model = AutoModelForCausalLM.from_pretrained(CFG["model_dir"], dtype=DTYPE).cuda().eval()
for _p in model.parameters():
    _p.requires_grad_(False)
IS_QWEN = "qwen" in CFG["student"].lower()

# ------------------------------------------------------------------ accounting
ACC = {"fwd_tokens": 0, "bwd_tokens": 0, "internal_fwd_tokens": 0}
_internal = [False]


def _emb_hook(m, i, o):
    n = int(o.shape[0] * o.shape[1])
    ACC["internal_fwd_tokens" if _internal[0] else "fwd_tokens"] += n
    if not _internal[0] and ACC["fwd_tokens"] > CFG["fte_budget"]:
        raise RuntimeError(f"student compute budget exhausted ({CFG['fte_budget']} forward tokens). Commit or discard, then call finish.")


def _bwd_hook(m, gi, go):
    if go and go[0] is not None:
        ACC["bwd_tokens"] += int(go[0].shape[0] * go[0].shape[1])


model.get_input_embeddings().register_forward_hook(_emb_hook)
def _decoder_layers(m):
    """Text decoder layers for plain causal LMs (model.model.layers) and Gemma 3 multimodal checkpoints (4B+), which load as
    Gemma3ForConditionalGeneration with the text stack at model.model.language_model.layers."""
    inner = m.model
    return inner.language_model.layers if hasattr(inner, "language_model") else inner.layers


_decoder_layers(model)[-1].register_full_backward_hook(_bwd_hook)


@contextlib.contextmanager
def internal():
    _internal[0] = True
    try:
        yield
    finally:
        _internal[0] = False


def render(user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    kw = {"enable_thinking": False} if IS_QWEN else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


@torch.no_grad()
def _generate(texts, max_new_tokens, bs=16):
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i])); out = [None] * len(texts)
    s = 0
    while s < len(order):
        idx = order[s:s + bs]
        try:
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2); continue
        for i, o in zip(idx, tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
            out[i] = o
        s += len(idx)
    return out


# ------------------------------------------------------------------ memory cap (1.5x inference peak)
with internal(), torch.no_grad():
    torch.cuda.reset_peak_memory_stats()
    model(torch.randint(1000, 5000, (1, 1024), device="cuda"))   # reference inference workload: batch 1, 1024 tokens
    INFER_PEAK = torch.cuda.max_memory_allocated()
MEM_CAP = int(CFG["mem_ratio"] * INFER_PEAK)
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(min(1.0, MEM_CAP * 1.08 / torch.cuda.get_device_properties(0).total_memory))

# ------------------------------------------------------------------ weight state
class _LazyBase:
    """Original weights, read on demand from the model's own (bf16) weight files and cast to float32, which is exactly
    what from_pretrained(dtype=float32) produced. Nothing is kept in RAM except tensors that fail the start-up check."""

    def __init__(self):
        import glob
        from safetensors import safe_open
        self.where, self.ram = {}, {}
        self.handles = [safe_open(f, "pt") for f in sorted(glob.glob(os.path.join(CFG["model_dir"], "*.safetensors")))]
        for h in self.handles:
            for k in h.keys():
                self.where[k] = h
        # parameter name -> key in the weight files (Gemma 3 multimodal files use the older 'language_model.model.*' layout)
        alias = lambda n: next((k for k in (n, n.replace("model.language_model.", "language_model.model.", 1),
                                            n.replace("model.vision_tower.", "vision_tower.", 1),
                                            n.replace("model.multi_modal_projector.", "multi_modal_projector.", 1)) if k in self.where), None)
        self.key = {}
        for n, p in model.named_parameters():
            cur = p.detach().cpu(); k = alias(n)
            if k is not None:
                self.key[n] = k
            if k is None or not torch.equal(self.where[k].get_tensor(k).to(cur.dtype), cur):
                self.ram[n] = cur.clone()
        self.names = [n for n, _ in model.named_parameters()]

    def __contains__(self, n):
        return n in self.ram or n in self.key

    def get(self, n, default=None):
        if n in self.ram:
            return self.ram[n]
        if n in self.key:
            k = self.key[n]; return self.where[k].get_tensor(k).to(DTYPE)
        return default

    def __getitem__(self, n):
        return self.get(n)


BASE = _LazyBase()
print("lazy base: kept in RAM", len(BASE.ram), "of", len(BASE.names), file=sys.stderr)
COMMITTED = {}         # overlay: tensors that differ from BASE in the committed state (and added bias vectors)
EXTRA_SITES = {}       # param name -> module path, for bias parameters added by the teacher
_hooks = {}
EDITS = []             # accepted edit ids


def _module(path):
    m = model
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def _add_bias(site):
    key = "tdwa_bias__" + site.replace(".", "__")
    if key in dict(model.named_parameters()):
        return key
    mod = _module(site)
    p = torch.nn.Parameter(torch.zeros(model.config.hidden_size, device="cuda", dtype=DTYPE), requires_grad=False)
    model.register_parameter(key, p)

    def hook(m, i, o):
        if isinstance(o, tuple):
            return (o[0] + p,) + tuple(o[1:])
        return o + p
    _hooks[key] = mod.register_forward_hook(hook)
    EXTRA_SITES[key] = site
    return key


def _ref(n, state=None):
    state = COMMITTED if state is None else state
    t = state.get(n)
    return BASE.get(n) if t is None else t


def _diff():
    out = {}
    for n, p in model.named_parameters():      # one tensor at a time: no second full copy of the model in RAM
        cur = p.detach().cpu()
        ref = _ref(n)
        if ref is None:
            ref = torch.zeros_like(cur)
        if not torch.equal(cur, ref):
            out[n] = (ref, cur)
    return out


def _load_state(state):
    for n, p in model.named_parameters():
        ref = _ref(n, state)
        if ref is None:
            p.data.zero_()
        else:
            p.data.copy_(ref.to("cuda"))


def _apply_delta_file(state, path):
    d = load_file(path)
    for k in d:
        n, kind = k.rsplit("::", 1)
        if kind == "dense_new":
            state[n] = d[k].clone()
        elif kind == "idx":
            t = _ref(n, state)
            if t is None:
                t = torch.zeros(tuple(json.load(open(path.replace("delta.safetensors", "meta.json")))["shapes"][n]))
            t = t.clone(); t.view(-1)[d[k]] = d[n + "::new"]; state[n] = t
    return state


# ------------------------------------------------------------------ regression guard
GENERIC = open(os.path.join(CFG["data_dir"], "generic.txt")).read()
CANARY = [json.loads(l) for l in open(os.path.join(CFG["data_dir"], "canary.jsonl"))]


@torch.no_grad()
def _regression():
    with internal():
        ids = tok(GENERIC, return_tensors="pt").input_ids.cuda()
        ppl = math.exp(model(ids, labels=ids).loss.item())
        mc = [c for c in CANARY if c["kind"] == "mc"]; ar = [c for c in CANARY if c["kind"] == "arith"]
        texts = [render(c["prompt"]) + "(" for c in mc]
        ok = 0
        for s in range(0, len(texts), 16):
            enc = tok(texts[s:s + 16], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            lg = model(**enc).logits[:, -1]
            for j, c in enumerate(mc[s:s + 16]):
                letters = [tok.encode(x, add_special_tokens=False)[0] for x in "ABCDE"[:c["n"]]]
                ok += "ABCDE"[int(lg[j, letters].argmax())] == c["answer"]
        outs = _generate([render(c["prompt"]) for c in ar], 12)
        ok2 = sum(c["answer"] in o.replace(",", "") for c, o in zip(ar, outs))
    return {"ppl": round(ppl, 4), "canary_mc": round(ok / len(mc), 4), "canary_arith": round(ok2 / len(ar), 4),
            "canary": round((ok + ok2) / len(CANARY), 4)}


REG_BASE = _regression()
LEDGER = open(os.path.join(RUN, "ledger.jsonl"), "a")


def _ledger(**kw):
    LEDGER.write(json.dumps({"t": time.time(), **kw}, default=str) + "\n"); LEDGER.flush()


# ------------------------------------------------------------------ StudentAPI
class StudentAPI:
    """Everything the teacher may use. `S.model` / `S.tok` are the raw HF model (float32, or bfloat16 for large students; on cuda) and tokenizer."""
    model = model
    tok = tok
    op_classes = OP_CLASSES
    task = CFG["task"]
    dev = [json.loads(l) for l in open(os.path.join(CFG["data_dir"], CFG["task"], "dev.jsonl"))]

    def info(self):
        """Budgets, limits and current state."""
        return {"student": CFG["student"], "task": CFG["task"], "n_dev": len(self.dev),
                "forward_tokens_used": ACC["fwd_tokens"], "backward_tokens_used": ACC["bwd_tokens"], "forward_token_budget": CFG["fte_budget"],
                "memory_cap_mib": round(MEM_CAP / 2**20), "inference_peak_mib": round(INFER_PEAK / 2**20),
                "accepted_edits": list(EDITS), "uncommitted_tensors": sorted(_diff()), "regression_baseline": REG_BASE,
                "regression_limits": {"ppl_max": round(REG_BASE["ppl"] * 1.10, 3), "canary_min": round(REG_BASE["canary"] - 0.03, 4)}}

    def render(self, user, system=None):
        """Chat-template a user message into the exact text fed to the student."""
        return render(user, system)

    def format_prompt(self, item, mode="free"):
        """The exact user message used in quizzes, official validation and the final hidden test."""
        return G.format_prompt(item, mode)

    def chat(self, users, max_new_tokens=512, system=None, batch_size=16):
        """Greedy generation for a list of user messages. Returns a list of strings."""
        return _generate([render(u, system) for u in ([users] if isinstance(users, str) else users)], max_new_tokens, batch_size)

    def grade(self, item, output, mode="free"):
        return G.grade(item, output, mode)

    def quiz(self, items, label, skill="", max_new_tokens=1024, mode="free", source="dev"):
        """Run items (dicts with task/question/answer, e.g. S.dev entries or your own) in the official prompt format,
        grade them, and log every item to quizzes/. Returns per-item dicts with output, pred, correct."""
        outs = _generate([render(G.format_prompt(it, mode)) for it in items], max_new_tokens)
        n = len(os.listdir(os.path.join(RUN, "quizzes"))) + 1
        rows = []
        with open(os.path.join(RUN, "quizzes", f"{n:03d}.jsonl"), "w") as f:
            for it, o in zip(items, outs):
                r = {"quiz": n, "label": label, "skill": skill, "source": source, "mode": mode, "after_edit": EDITS[-1] if EDITS else 0,
                     "uncommitted": bool(_diff()), "id": it.get("id"), "group": it.get("group"), "question": it["question"],
                     "answer": it["answer"], "output": o, **G.grade(it, o, mode)}
                rows.append(r); f.write(json.dumps(r, ensure_ascii=False) + "\n")
        acc = sum(r["correct"] for r in rows) / max(1, len(rows))
        _ledger(event="quiz", quiz=n, label=label, n=len(rows), acc=acc)
        print(f"[quiz {n:03d} '{label}'] n={len(rows)} acc={acc:.3f}")
        return rows

    def add_bias(self, site):
        """Add a zero-initialised bias vector (size hidden) to the output of the module at `site`,
        e.g. 'model.layers.13', 'model.layers.13.mlp'. Returns the new parameter name (edit it via S.params()[name])."""
        return _add_bias(site)

    def params(self):
        """dict name -> parameter tensor (edit in place with .data / copy_). Includes added bias vectors."""
        return dict(model.named_parameters())

    def regression_check(self):
        """Perplexity on fixed generic text + canary accuracy for the CURRENT weights (does not use your budget)."""
        return _regression()

    def discard(self):
        """Throw away all uncommitted weight changes."""
        _load_state(COMMITTED); return "reverted to last committed state"

    def commit_edit(self, meta):
        """Register the current uncommitted weight changes as one edit. `meta` must contain:
        hypothesis, method, op_class (one of S.op_classes), prediction, data_used. The harness saves every changed
        position with old and new values, runs the regression guard, and REJECTS + reverts the edit if the guard fails."""
        missing = [k for k in REQUIRED_META if k not in meta]
        if missing:
            raise ValueError(f"meta is missing {missing}")
        if meta["op_class"] not in OP_CLASSES:
            raise ValueError(f"op_class must be one of {OP_CLASSES}")
        diff = _diff()
        if not diff:
            raise ValueError("no uncommitted weight changes")
        eid = len(os.listdir(os.path.join(RUN, "edits"))) + 1
        d = os.path.join(RUN, "edits", f"{eid:03d}"); os.makedirs(d)
        tensors, targets, shapes = {}, [], {}
        for n, (ref, cur) in diff.items():
            delta = cur - ref; nz = delta != 0; cnt = int(nz.sum())
            t = {"param": n, "shape": list(cur.shape), "n_changed": cnt, "frob_before": float(ref.norm()), "frob_delta": float(delta.norm()),
                 "rel_norm": float(delta.norm() / (ref.norm() + 1e-12)), "max_abs_delta": float(delta.abs().max()),
                 "new_parameter": n in EXTRA_SITES and n not in BASE, "site": EXTRA_SITES.get(n)}
            if delta.dim() == 2 and delta.numel() <= 2e7:
                sv = torch.linalg.svdvals(delta.cuda()).cpu()
                t["rank"] = int((sv > 1e-4 * sv[0]).sum()); t["top_singular_values"] = [round(float(x), 6) for x in sv[:5]]
            if cnt <= 0.2 * delta.numel():
                idx = nz.view(-1).nonzero().squeeze(1)
                tensors[n + "::idx"] = idx; tensors[n + "::old"] = ref.reshape(-1)[idx].clone(); tensors[n + "::new"] = cur.reshape(-1)[idx].clone()
                t["storage"] = "sparse (index, old value, new value)"
            else:
                tensors[n + "::dense_new"] = cur.clone(); t["storage"] = "dense new tensor (old values = state before this edit)"
            shapes[n] = list(cur.shape); targets.append(t)
        save_file(tensors, os.path.join(d, "delta.safetensors"))
        reg = _regression()
        ok = reg["ppl"] <= REG_BASE["ppl"] * 1.10 and reg["canary"] >= REG_BASE["canary"] - 0.03
        rec = {"edit_id": eid, "parent_edit": EDITS[-1] if EDITS else 0, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **{k: meta[k] for k in meta},
               "targets": targets, "shapes": shapes, "n_values_changed": sum(t["n_changed"] for t in targets),
               "cost_so_far": dict(ACC), "regression_before": getattr(self, "_last_reg", REG_BASE), "regression_after": reg,
               "regression_baseline": REG_BASE, "accepted": bool(ok)}
        json.dump(rec, open(os.path.join(d, "meta.json"), "w"), indent=1, ensure_ascii=False, default=str)
        _ledger(event="commit", edit=eid, accepted=ok, regression=reg, n_values=rec["n_values_changed"])
        if ok:
            for n, (ref, cur) in diff.items():
                COMMITTED[n] = cur.clone()
            EDITS.append(eid); self._last_reg = reg; _save_state()
            return {"edit_id": eid, "accepted": True, "regression": reg, "n_values_changed": rec["n_values_changed"]}
        _load_state(COMMITTED)
        return {"edit_id": eid, "accepted": False, "reason": "regression guard failed; edit reverted", "regression": reg,
                "limits": self.info()["regression_limits"]}

    def rollback(self, to_edit_id, reason):
        """Return to the state right after accepted edit `to_edit_id` (0 = original weights). Logged."""
        keep = [e for e in EDITS if e <= to_edit_id]
        state = {}
        for e in keep:
            state = _apply_delta_file(state, os.path.join(RUN, "edits", f"{e:03d}", "delta.safetensors"))
        for n in EXTRA_SITES:
            state.setdefault(n, torch.zeros(model.config.hidden_size))
        COMMITTED.clear(); COMMITTED.update(state); _load_state(COMMITTED)
        removed = [e for e in EDITS if e > to_edit_id]; EDITS[:] = keep
        _ledger(event="rollback", to=to_edit_id, removed=removed, reason=reason); _save_state()
        return {"now_at_edit": to_edit_id, "removed": removed}


S = StudentAPI()

# ------------------------------------------------------------------ crash recovery
# The kernel can be killed by teacher code (a CUDA device-side assert poisons the process; a call can time out).
# The parent restarts it; accepted edits, added bias sites and the cost counters are restored from disk.
# Uncommitted changes and the teacher's Python variables are lost.
STATE_PATH = os.path.join(RUN, "kstate.json")


def _save_state():
    json.dump({"edits": EDITS, "extra_sites": EXTRA_SITES, "acc": ACC, "last_reg": getattr(S, "_last_reg", None)}, open(STATE_PATH + ".tmp", "w"))
    os.replace(STATE_PATH + ".tmp", STATE_PATH)


RESUMED = os.path.exists(STATE_PATH)
if RESUMED:
    _st = json.load(open(STATE_PATH))
    for _site in _st["extra_sites"].values():
        _add_bias(_site)
    _state = {}
    for _e in _st["edits"]:
        _state = _apply_delta_file(_state, os.path.join(RUN, "edits", f"{_e:03d}", "delta.safetensors"))
    for _n in EXTRA_SITES:
        _state.setdefault(_n, torch.zeros(model.config.hidden_size))
    COMMITTED.update(_state); _load_state(COMMITTED); EDITS[:] = _st["edits"]; ACC.update(_st["acc"])
    if _st.get("last_reg"):
        S._last_reg = _st["last_reg"]
    _ledger(event="kernel_restart", edits=list(EDITS), acc=dict(ACC))
MAXC = CFG.get("max_output_chars", 8000)


def trunc(s):
    return s if len(s) <= MAXC else s[:MAXC // 2] + f"\n...[truncated {len(s) - MAXC} chars]...\n" + s[-MAXC // 2:]


ns = {"S": S, "torch": torch, "np": np, "json": json, "__name__": "__tdwa__"}
os.chdir(WORK)
os.environ.pop("TDWA_CFG", None)
_OUT.write(json.dumps({"ready": True, "resumed": RESUMED, "info": S.info()}) + "\n"); _OUT.flush()

for line in sys.stdin:
    req = json.loads(line)
    if req.get("cmd") == "finish":
        dropped = sorted(_diff()); _load_state(COMMITTED)
        final = {n: t for n, t in COMMITTED.items() if n not in BASE or not torch.equal(t, BASE[n])}
        os.makedirs(os.path.join(RUN, "final"), exist_ok=True)
        save_file({n: t.contiguous() for n, t in final.items()} or {"_empty": torch.zeros(1)}, os.path.join(RUN, "final", "edit.safetensors"))
        json.dump({"accepted_edits": EDITS, "changed_tensors": sorted(final), "extra_sites": EXTRA_SITES, "dropped_uncommitted": dropped,
                   "values_changed_vs_base": int(sum((t != BASE[n]).sum() if n in BASE else t.numel() for n, t in final.items())),
                   "cost": dict(ACC), "regression_final": _regression(), "regression_baseline": REG_BASE,
                   "inference_peak_mib": INFER_PEAK / 2**20, "mem_cap_mib": MEM_CAP / 2**20},
                  open(os.path.join(RUN, "final", "summary.json"), "w"), indent=1, default=str)
        _OUT.write(json.dumps({"ok": True, "accepted_edits": EDITS}) + "\n"); _OUT.flush()
        break
    if req.get("cmd") == "generate":
        if _diff():
            resp = {"ok": False, "error": "uncommitted weight changes present; commit_edit or discard first"}
        else:
            with internal():
                outs = _generate([render(p) for p in req["prompts"]], req["max_new_tokens"])
            resp = {"ok": True, "outputs": outs, "after_edit": EDITS[-1] if EDITS else 0}
        _OUT.write(json.dumps(resp) + "\n"); _OUT.flush(); continue
    out, err = io.StringIO(), io.StringIO(); ok = True
    f0, b0 = ACC["fwd_tokens"], ACC["bwd_tokens"]; t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            exec(compile(req["code"], "<teacher>", "exec"), ns)
        except BaseException:
            ok = False; traceback.print_exc(file=err)
    try:
        peak = torch.cuda.max_memory_allocated(); torch.zeros(1, device="cuda").item(); unc = bool(_diff())
    except RuntimeError as e:                    # CUDA context is unusable; the parent restarts the kernel
        ACC["fwd_tokens"] = max(ACC["fwd_tokens"], f0); _save_state()
        _ledger(event="kernel_fatal", call=req.get("call"), error=str(e)[:300], teacher_stderr=err.getvalue()[-1500:])
        sys.stderr.write("FATAL CUDA state after teacher code:\n" + err.getvalue()[-3000:] + "\n"); os._exit(3)
    stat = {"fwd_tokens": ACC["fwd_tokens"] - f0, "bwd_tokens": ACC["bwd_tokens"] - b0, "peak_mem_ratio": round(peak / INFER_PEAK, 3),
            "sec": round(time.time() - t0, 1), "uncommitted": unc,
            "forward_tokens_left": CFG["fte_budget"] - ACC["fwd_tokens"]}
    _ledger(event="exec", call=req.get("call"), ok=ok, **stat); _save_state()
    _OUT.write(json.dumps({"ok": ok, "stdout": trunc(out.getvalue()), "stderr": trunc(err.getvalue()), "stat": stat}) + "\n")
    _OUT.flush()
