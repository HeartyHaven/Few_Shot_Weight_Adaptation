"""Where did each teacher edit each student?  Builds report/site/atlas.json from the saved edit records.

For every run and every committed edit (kept, rolled back, or rejected by the guard) it reconstructs the exact
weight change (sparse records directly; dense records by replaying the run's edit/rollback history from the
original weights) and breaks it down into model units:
  attention q/k/v_proj rows and o_proj columns  -> heads (KV heads for k/v)
  mlp gate/up_proj rows and down_proj columns   -> neurons
  anything else                                 -> the whole tensor
Each unit gets the Frobenius norm of its change. Usage: python report/build_atlas.py
"""
import glob, json, os, re, sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file

D = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(D)
HF = "/mnt/data2/users/lang.gao/huggingface_cache"
TOPK = 24


def model_files(repo):
    snap = glob.glob(os.path.join(HF, "models--" + repo.replace("/", "--"), "snapshots", "*"))[0]
    cfg = json.load(open(os.path.join(snap, "config.json")))
    cfg = cfg.get("text_config", cfg)
    where = {}
    for f in glob.glob(os.path.join(snap, "*.safetensors")):
        h = safe_open(f, "pt")
        for k in h.keys():
            where[k] = h
    return cfg, where


def base_tensor(where, name):
    for k in (name, name.replace("model.", "model.language_model.", 1)):
        if k in where:
            return where[k].get_tensor(k).float()
    return None


PAT = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.weight$")


def units(name, delta, cfg):
    """Split a weight change into named units. Returns (layer, module, unit_kind, [(unit_index, norm)])."""
    m = PAT.match(name)
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    if not m:
        if name.startswith("tdwa_bias__"):
            site = name[len("tdwa_bias__"):].replace("__", ".")
            lm = re.search(r"layers\.(\d+)", site)
            return (int(lm.group(1)) if lm else None, "bias:" + site.split(".")[-1], "vector", [(0, float(delta.norm()))])
        return (None, name, "tensor", [(0, float(delta.norm()))])
    layer, block, mod = int(m.group(1)), m.group(2), m.group(3)
    if mod in ("q_proj", "k_proj", "v_proj"):
        n = delta.shape[0] // hd
        return (layer, mod, "head", [(i, float(delta[i * hd:(i + 1) * hd].norm())) for i in range(n)])
    if mod == "o_proj":
        n = delta.shape[1] // hd
        return (layer, mod, "head", [(i, float(delta[:, i * hd:(i + 1) * hd].norm())) for i in range(n)])
    if mod in ("gate_proj", "up_proj"):
        return (layer, mod, "neuron", list(enumerate(delta.norm(dim=1).tolist())))
    if mod == "down_proj":
        return (layer, mod, "neuron", list(enumerate(delta.norm(dim=0).tolist())))
    return (layer, mod, "tensor", [(0, float(delta.norm()))])


def run_atlas(R):
    name = os.path.basename(R); cfgr = json.load(open(R + "/config.json"))
    metas = {json.load(open(m))["edit_id"]: json.load(open(m)) for m in glob.glob(R + "/edits/*/meta.json")}
    if not metas:
        return None
    cfg, where = model_files(cfgr["student"])
    led = [json.loads(l) for l in open(R + "/ledger.jsonl")]
    tr = [json.loads(l) for l in open(R + "/transcript.jsonl")]
    tools = [e for e in tr if e.get("event") == "tool"]; t0 = tr[0]["t"]
    exams = [json.load(open(x)) for x in sorted(glob.glob(R + "/exams/*.json"))]
    base_cache, committed, accepted = {}, {}, []

    def base(n):
        if n not in base_cache:
            t = base_tensor(where, n)
            base_cache[n] = t
        return base_cache[n]

    def cur(n, shape=None):
        if n in committed:
            return committed[n]
        b = base(n)
        return b if b is not None else torch.zeros(shape)

    def apply(state, eid):
        d = load_file(os.path.join(R, "edits", f"{eid:03d}", "delta.safetensors"))
        shapes = metas[eid].get("shapes") or {t["param"]: t["shape"] for t in metas[eid]["targets"]}
        new = {}
        for k in d:
            n, kind = k.rsplit("::", 1)
            if kind == "dense_new":
                new[n] = d[k].float()
            elif kind == "idx":
                t = (state[n] if n in state else cur(n, shapes.get(n))).clone().reshape(-1)
                t[d[k]] = d[n + "::new"].float(); new[n] = t.reshape(shapes[n])
        return new

    edits = []
    for ev in led:
        if ev["event"] == "rollback":
            committed.clear()
            for e in [e for e in accepted if e <= ev["to"]]:
                committed.update(apply(committed, e))
            accepted[:] = [e for e in accepted if e <= ev["to"]]
            for x in edits:
                if x["id"] in ev["removed"]:
                    x["rolled_back"] = True; x["rolled_back_reason"] = ev.get("reason", "")
            continue
        if ev["event"] != "commit":
            continue
        eid = ev["edit"]; meta = metas[eid]
        new = apply(committed, eid)
        step = next((i + 1 for i, e in enumerate(tools) if e["t"] >= ev["t"] - 1), len(tools))
        targets = []
        for n, tnew in new.items():
            old = cur(n, list(tnew.shape))
            delta = tnew - old
            layer, mod, kind, us = units(n, delta, cfg)
            tot = sum(v * v for _, v in us) ** 0.5 or 1e-12
            nz = [(i, v) for i, v in us if v > 1e-9]
            top = sorted(nz, key=lambda x: -x[1])[:TOPK]
            mt = next((t for t in meta["targets"] if t["param"] == n), {})
            targets.append({"param": n, "layer": layer, "module": mod, "unit_kind": kind, "n_units": len(us), "n_units_changed": len(nz),
                            "units": [[i, round(v / tot, 4)] for i, v in top],
                            "top_share": round(sum(v * v for _, v in top[:4]) / tot ** 2, 4) if nz else 0,
                            "rel_norm": round(mt.get("rel_norm", float(delta.norm() / (old.norm() + 1e-12))), 4),
                            "n_values": int((delta != 0).sum()), "rank": mt.get("rank"), "top_sv": (mt.get("top_singular_values") or [])[:3]})
        after = [round(x["accuracy"] * 100, 1) for x in exams if x["after_edit"] == eid]
        x = {"id": eid, "step": step, "minute": round((ev["t"] - t0) / 60, 1), "accepted": bool(ev["accepted"]), "rolled_back": False,
             "op_class": meta.get("op_class"), "n_values": meta.get("n_values_changed"), "val_after": after[0] if after else None,
             "hypothesis": meta.get("hypothesis"), "method": meta.get("method"), "prediction": meta.get("prediction"),
             "data_used": meta["data_used"] if isinstance(meta.get("data_used"), str) else json.dumps(meta.get("data_used"), ensure_ascii=False),
             "regression": meta.get("regression_after"), "targets": targets}
        edits.append(x)
        if ev["accepted"]:
            committed.update(new); accepted.append(eid)
    kept = set(accepted)
    for x in edits:
        x["kept_final"] = x["id"] in kept
    return {"run": name, "task": cfgr["task"], "student": cfgr["student"], "teacher": cfgr["teacher"], "seed": cfgr["seed"],
            "arch": {"layers": cfg["num_hidden_layers"], "heads": cfg["num_attention_heads"], "kv_heads": cfg.get("num_key_value_heads"),
                     "head_dim": cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"], "hidden": cfg["hidden_size"],
                     "mlp": cfg["intermediate_size"], "layer_types": cfg.get("layer_types")},
            "n_steps": len(tools), "exams": [[x["after_edit"], round(x["accuracy"] * 100, 1)] for x in exams], "edits": edits}


def main():
    only = sys.argv[1:]
    out = []
    for R in sorted(glob.glob(ROOT + "/runs/*")):
        if only and not any(o in R for o in only):
            continue
        try:
            a = run_atlas(R)
        except Exception as e:
            print("FAILED", os.path.basename(R), type(e).__name__, e); continue
        if a:
            out.append(a); print(os.path.basename(R), len(a["edits"]), "edits", flush=True)
    os.makedirs(D + "/site", exist_ok=True)
    json.dump({"runs": out}, open(D + "/site/atlas.json", "w"), ensure_ascii=False)
    print("wrote site/atlas.json", round(os.path.getsize(D + "/site/atlas.json") / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
