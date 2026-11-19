"""WP0: module structure, determinism, rank-one edit smoke test, and peak-memory ratios.

Usage: python arch_smoke.py <model_repo> <out_dir>

Smoke test question: if we add a rank-one update to one layer's MLP output projection so that the MLP
output moves by alpha * u (u a fixed unit direction) on a given prompt, how much of that reaches the
residual stream right after the block? A ratio near 1 means the write lands at full size; a ratio that
does not grow with alpha means a normalisation after the MLP is absorbing the magnitude.
"""
import json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO, OUT = sys.argv[1], sys.argv[2]
HF = os.environ["HF_HOME"]
tag = REPO.split("/")[-1]
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(REPO, cache_dir=HF)
model = AutoModelForCausalLM.from_pretrained(REPO, cache_dir=HF, dtype=torch.float32).cuda().eval()
res = {"model": REPO}

# ---- structure ----
layers = model.model.layers
with open(f"{OUT}/arch_{tag}.txt", "w") as f:
    f.write(str(layers[0]) + "\n\nchildren of layer 0: " + ", ".join(n for n, _ in layers[0].named_children()) + "\n")
    f.write(f"n_layers={len(layers)}\n")
    for n, p in layers[0].named_parameters():
        f.write(f"{n}\t{tuple(p.shape)}\n")
res["layer_children"] = [n for n, _ in layers[0].named_children()]
res["mlp_has_bias"] = layers[0].mlp.down_proj.bias is not None
res["n_layers"] = len(layers)

kw = {"enable_thinking": False} if "Qwen3" in REPO else {}
text = tok.apply_chat_template([{"role": "user", "content": "Alice has a red ball and Bob has a blue ball. They swap. "
                                 "What does Alice have now?"}], tokenize=False, add_generation_prompt=True, **kw)
enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")

# ---- determinism ----
with torch.no_grad():
    g1 = model.generate(**enc, max_new_tokens=40, do_sample=False)
    g2 = model.generate(**enc, max_new_tokens=40, do_sample=False)
res["greedy_deterministic"] = bool(torch.equal(g1, g2))
res["sample_output"] = tok.decode(g1[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

# ---- rank-one edit smoke test ----
L = len(layers) // 2
blk = layers[L]
cap = {}
h1 = blk.mlp.down_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("key", i[0][0, -1].detach().clone()) or
                                              cap.__setitem__("mlp_out", o[0, -1].detach().clone()))
h2 = blk.register_forward_hook(lambda m, i, o: cap.__setitem__("resid", (o[0] if isinstance(o, tuple) else o)[0, -1].detach().clone()))


@torch.no_grad()
def run():
    logits = model(**enc).logits[0, -1].float()
    return cap["key"].clone(), cap["mlp_out"].clone(), cap["resid"].clone(), torch.log_softmax(logits, -1)


k0, m0, r0, lp0 = run()
torch.manual_seed(0)
u = torch.randn_like(m0); u = u / u.norm()
W = blk.mlp.down_proj.weight
W0 = W.detach().clone()
rows = []
res["mlp_raw_out_norm"] = round(m0.norm().item(), 3)
res["resid_norm"] = round(r0.norm().item(), 3)
# two sweeps: write size relative to the residual norm, then relative to the raw MLP output norm
for base_name, base, rel in [("resid", r0.norm().item(), x) for x in (0.1, 0.5, 1.0, 2.0, 5.0)] + \
                           [("mlp_out", m0.norm().item(), x) for x in (0.1, 0.3, 1.0, 3.0, 10.0)]:
    alpha = rel * base
    with torch.no_grad():
        W.copy_(W0 + alpha * torch.outer(u, k0) / (k0 @ k0))   # rank-one: W k0 -> W k0 + alpha u
    k1, m1, r1, lp1 = run()
    dm, dr = m1 - m0, r1 - r0
    rows.append({"relative_to": base_name, "rel_requested": rel, "alpha": round(alpha, 3),
                 "mlp_out_delta_norm": round(dm.norm().item(), 3),
                 "resid_delta_norm": round(dr.norm().item(), 3),
                 "resid_delta_over_alpha": round(dr.norm().item() / alpha, 4),
                 "cos_resid_delta_u": round(torch.cosine_similarity(dr, u, dim=0).item(), 4),
                 "resid_norm_before": round(r0.norm().item(), 3), "resid_norm_after": round(r1.norm().item(), 3),
                 "kl_next_token": round(torch.sum(lp0.exp() * (lp0 - lp1)).item(), 5),
                 "frob_rel_weight_change": round(((W - W0).norm() / W0.norm()).item(), 5)})
    with torch.no_grad():
        W.copy_(W0)
_, _, r_back, _ = run()
res["smoke_layer"] = L
res["smoke"] = rows
res["restored_exactly"] = bool(torch.equal(r_back, r0))
h1.remove(); h2.remove()

# ---- peak memory: batch 1, 1024 tokens ----
model = model.to(torch.bfloat16)
ids = torch.randint(1000, 5000, (1, 1024), device="cuda")


def peak(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def infer():
    with torch.no_grad():
        model(ids)


def with_hooks():
    store, hs = [], []
    for b in layers:
        hs.append(b.register_forward_hook(lambda m, i, o: store.append((o[0] if isinstance(o, tuple) else o).detach())))
        hs.append(b.mlp.register_forward_hook(lambda m, i, o: store.append(o.detach())))
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()


def local_backprop():
    """ROME-style: optimise a vector added at one layer; gradients flow through later layers only to that vector."""
    for p in model.parameters():
        p.requires_grad_(False)
    v = torch.zeros(model.config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    h = layers[L].mlp.register_forward_hook(lambda m, i, o: o + v)
    loss = model(ids).logits[0, -1].float().log_softmax(-1)[0]
    loss.backward()
    h.remove()


base = peak(infer)
res["memory_mib"] = {"weights_only": round(torch.cuda.memory_allocated() / 2**20, 1), "inference_peak": round(base, 1)}
for name, fn in (("hooks_all_layers", with_hooks), ("local_backprop_one_vector", local_backprop)):
    p = peak(fn)
    res["memory_mib"][name] = round(p, 1)
    res["memory_mib"][name + "_ratio"] = round(p / base, 3)
json.dump(res, open(f"{OUT}/smoke_{tag}.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
