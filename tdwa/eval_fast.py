"""Fast hidden-test evaluation of a teaching run (or an unedited student) with vLLM in float32.

  python -m tdwa.eval_fast runs/<name> [--gpu 0] [--mem 0.28]
  python -m tdwa.eval_fast --base --task t1 --student Qwen/Qwen3-4B [--gpu 0]

Steps: merge the run's final edits into a full float32 model folder (tdwa.merge_run), evaluate it with
tdwa.evaluate_vllm (same prompts, graders, hidden tests and retention suite as tdwa.evaluate), write eval/<name>.json
with the same metadata fields as tdwa.evaluate, then delete the merged folder.

Agreement with tdwa.evaluate (HF float32), measured on 2026-11-19 (Gemma 1B, object tracking, 1200 items):
unedited student 98% identical outputs and 99.3-99.7% identical grading; an edited student (Astra 6.2 s0) gave the
same accuracy on two of three test sets and 83.3 vs 84.0 on the third. About 15x faster.
"""
import argparse, glob, json, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = os.environ.get("TDWA_HF", "/mnt/data2/users/lang.gao/huggingface_cache")   # per-host override (masdar: cache/hf)
VPY = os.path.join(ROOT, ".venv-vllm", "bin", "python")
PY = sys.executable
ENV = dict(os.environ, VLLM_ATTENTION_BACKEND="TRITON_ATTN", TDWA_VLLM_EAGER=os.environ.get("TDWA_VLLM_EAGER", "1"), VLLM_USE_FLASHINFER_SAMPLER="0", HF_HOME=HF,
           HF_HUB_CACHE=HF, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", VLLM_CACHE_ROOT=os.path.join(ROOT, "cache", "vllm"),
           VLLM_LOGGING_LEVEL="WARNING", TOKENIZERS_PARALLELISM="false",
           NCCL_P2P_DISABLE="1")      # this host's GPU peer copies are broken (they return zeros); NCCL goes through host memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?"); ap.add_argument("--base", action="store_true"); ap.add_argument("--task"); ap.add_argument("--student")
    ap.add_argument("--gpu", default="0"); ap.add_argument("--mem", default="0.28"); ap.add_argument("--keep", action="store_true")
    ap.add_argument("--b1", action="store_true", help="also put the rule/procedure text before each test question (B1 prompt); result name gets __b1")
    ap.add_argument("--dtype", default="float32", help="bfloat16 for 8B+ students (2026-11-23)")
    ap.add_argument("--tp", default="1", help="tensor parallel size (2 = split over --gpu 0,1)")
    ap.add_argument("--ret", action="store_true", help="2026-11-25 (RET-FULL): MMLU + GSM8K (tdwa.retention_vllm) instead of the hidden test; "
                    "writes eval_ret/<run>.json or eval_ret/base__<Student>__<vllm32|vllm16>.json; an unmergeable run gets eval_ret/<run>.skip")
    a = ap.parse_args()
    tag = "vllm32" if a.dtype == "float32" else "vllm16"
    env = dict(ENV, CUDA_VISIBLE_DEVICES=str(a.gpu))
    if a.base and a.ret:
        snap = glob.glob(os.path.join(HF, "models--" + a.student.replace("/", "--"), "snapshots", "*"))[0]
        cmd = [VPY, "-m", "tdwa.retention_vllm", "--student", a.student, "--model-path", snap, "--dtype", a.dtype, "--mem", a.mem,
               "--name", f"base__{a.student.split('/')[-1]}__{tag}"]
        sys.exit(subprocess.call(cmd, cwd=ROOT, env=env))
    if a.base:
        snap = glob.glob(os.path.join(HF, "models--" + a.student.replace("/", "--"), "snapshots", "*"))[0]
        name = f"base__{a.task}__{a.student.split('/')[-1]}__{tag}"
        cmd = [VPY, "-m", "tdwa.evaluate_vllm", "--task", a.task, "--student", a.student, "--model-path", snap, "--dtype", a.dtype,
               "--mem", a.mem, "--tp", str(a.tp), "--name", name + ("__b1" if a.b1 else "")] + (["--b1"] if a.b1 else [])
        sys.exit(subprocess.call(cmd, cwd=ROOT, env=env))
    run = os.path.abspath(a.run.rstrip("/")); name = os.path.basename(run)
    cfg = json.load(open(os.path.join(run, "config.json")))
    merged = os.path.join(ROOT, "cache", "merged", name + ("__ret" if a.ret else ""))   # own copy: a hidden test of the same run may be merging too
    r = subprocess.call([PY, "-m", "tdwa.merge_run", run, "--out", merged, "--dtype", a.dtype], cwd=ROOT, env=env)
    if r:
        if a.ret:
            os.makedirs(os.path.join(ROOT, "eval_ret"), exist_ok=True)
            open(os.path.join(ROOT, "eval_ret", name + ".skip"), "w").write("merge failed (e.g. a kept non-zero added bias); MMLU/GSM8K not measured\n")
        shutil.rmtree(merged, ignore_errors=True)
        sys.exit("merge failed; evaluate this run with tdwa.evaluate")
    if a.ret:
        cmd = [VPY, "-m", "tdwa.retention_vllm", "--student", cfg["student"], "--model-path", merged, "--dtype", a.dtype, "--mem", a.mem,
               "--name", name, "--meta", json.dumps({"teacher": cfg.get("teacher_tag") or cfg["teacher"], "task": cfg["task"]})]
        r = subprocess.call(cmd, cwd=ROOT, env=env)
        shutil.rmtree(merged, ignore_errors=True)
        sys.exit(r)
    summ = json.load(open(os.path.join(run, "final", "summary.json")))
    meta = {"teacher": cfg["teacher"], "values_changed": summ.get("values_changed_vs_base"), "run_summary": summ,
            "teacher_usage": cfg.get("teacher_usage"), "turns": cfg.get("turns"), "wall_seconds": cfg.get("wall_seconds"),
            "validation_looks": cfg.get("validation_looks")}
    cmd = [VPY, "-m", "tdwa.evaluate_vllm", "--task", cfg["task"], "--student", cfg["student"], "--model-path", merged, "--dtype", a.dtype,
           "--mem", a.mem, "--tp", str(a.tp), "--name", name + ("__b1" if a.b1 else ""), "--meta", json.dumps(meta, default=str)] + (["--b1"] if a.b1 else [])
    r = subprocess.call(cmd, cwd=ROOT, env=env)
    if not a.keep:
        shutil.rmtree(merged, ignore_errors=True)
    sys.exit(r)


if __name__ == "__main__":
    main()
