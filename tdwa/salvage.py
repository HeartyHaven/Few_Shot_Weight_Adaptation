"""Finalize a run whose teacher process died before calling finish (e.g. API quota). Restarts the sandbox, which restores
the committed edits from disk, and writes final/. Usage: python -m tdwa.salvage runs/<name> --gpu 0 --reason "..." """
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tdwa.teach import HF, Sandbox
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--gpu", default="0"); ap.add_argument("--reason", required=True); a = ap.parse_args()
run = os.path.abspath(a.run); cfg = json.load(open(run + "/config.json"))
assert not os.path.exists(run + "/final/summary.json"), "already finalized"
sb = Sandbox(cfg, run, os.path.join(HF, "models--" + cfg["student"].replace("/", "--")), a.gpu, 1200)
print("ready", sb.ready.get("resumed"), sb.ready["info"]["accepted_edits"])
fin = sb.rpc({"cmd": "finish"}, timeout=1800); sb.close()
os.makedirs(run + "/final", exist_ok=True)
open(run + "/final/report.md", "w").write(f"(no final report: the teacher process stopped before finishing. Reason: {a.reason}. Finalized by tdwa.salvage with the edits committed up to that point.)")
calls = sum(1 for l in open(run + "/transcript.jsonl") if '"event": "tool"' in l)
cfg.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), turns=calls, finish=fin, report_submitted=False, interrupted=a.reason,
           validation_looks=len(os.listdir(run + "/exams")))
json.dump(cfg, open(run + "/config.json", "w"), indent=1); print(fin)
