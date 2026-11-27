"""Write a teaching run's final student as a plain Hugging Face model directory (for vLLM).

  python -m tdwa.merge_run runs/<name> [--out cache/merged/<name>] [--dtype float32]

The directory holds the original config/tokenizer files plus one safetensors file with every original tensor,
replaced by the run's final edited tensors (runs/<name>/final/edit.safetensors). float32 is the default so that small
edits are not rounded away (bfloat16 keeps only ~3 significant digits). Added bias vectors cannot be expressed in
the original architecture: the script refuses if a run kept a non-zero one (evaluate such runs with tdwa.evaluate).
"""
import argparse, glob, json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = os.environ.get("TDWA_HF", "/mnt/data2/users/lang.gao/huggingface_cache")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--out"); ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    run = os.path.abspath(a.run); name = os.path.basename(run.rstrip("/"))
    cfg = json.load(open(os.path.join(run, "config.json")))
    snap = glob.glob(os.path.join(HF, "models--" + cfg["student"].replace("/", "--"), "snapshots", "*"))[0]
    out = a.out or os.path.join(ROOT, "cache", "merged", name)
    edit = load_file(os.path.join(run, "final", "edit.safetensors")); edit.pop("_empty", None)
    bias = {k: v for k, v in edit.items() if k.startswith("tdwa_bias__")}
    if any(bool(v.abs().sum()) for v in bias.values()):
        sys.exit(f"{name} kept a non-zero added bias ({list(bias)}); vLLM cannot load it. Use tdwa.evaluate.")
    edit = {k: v for k, v in edit.items() if k not in bias}
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(snap):
        if not f.endswith(".safetensors") and not os.path.isdir(os.path.join(snap, f)):   # config, tokenizer, shard index; 2026-11-25: skip
            # sub-directories (Llama 3.1 snapshots have original/ with Meta's consolidated checkpoint; copying it failed the merge)
            shutil.copy(os.path.realpath(os.path.join(snap, f)), os.path.join(out, f))
    # 2026-11-25: the kernel names parameters as the loaded model does (transformers 4.57: Gemma3ForConditionalGeneration ->
    # model.language_model.*, lm_head.*); the checkpoint files may use the older layout (language_model.model.*,
    # language_model.lm_head.*). Map every edited name to the key that exists in the checkpoint.
    keyset = set()
    for f in glob.glob(os.path.join(snap, "*.safetensors")):
        with safe_open(f, "pt") as h:
            keyset.update(h.keys())

    def _cands(k):
        out = [k]
        for new, old in (("model.language_model.", "language_model.model."), ("model.vision_tower.", "vision_tower."),
                         ("model.multi_modal_projector.", "multi_modal_projector."), ("lm_head.", "language_model.lm_head.")):
            if k.startswith(new):
                out.append(old + k[len(new):])
            if k.startswith(old):
                out.append(new + k[len(old):])
        return out
    edit = {next((c for c in _cands(k) if c in keyset), k): v for k, v in edit.items()}
    dt = getattr(torch, a.dtype); used = set(); n = 0
    for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):   # one shard at a time keeps peak RAM at one shard
        tensors = {}
        with safe_open(f, "pt") as h:
            for k in h.keys():
                t = edit[k] if k in edit else h.get_tensor(k)
                used.add(k) if k in edit else None
                tensors[k] = t.to(dt).contiguous()
        save_file(tensors, os.path.join(out, os.path.basename(f)), metadata={"format": "pt"}); n += len(tensors); del tensors
    missing = set(edit) - used
    if missing:
        sys.exit(f"edited tensors not found in the checkpoint: {sorted(missing)[:5]}")
    tensors = range(n)
    c = json.load(open(os.path.join(out, "config.json")))
    c["torch_dtype"] = a.dtype
    if "text_config" in c:
        c["text_config"]["torch_dtype"] = a.dtype
    json.dump(c, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(out, len(tensors), "tensors,", len(edit), "edited")


if __name__ == "__main__":
    main()
