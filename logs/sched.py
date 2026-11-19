"""Tiny GPU packer for this host. Jobs come from logs/jobs_<host>.jsonl (re-read every loop, so jobs can be appended):
  {"id": "...", "gpus": 1|2, "mem": <GB reserved per GPU>, "cmd": "<shell, run from project root>",
   "after": ["id", ...], "wait_file": "<path that must exist first>"}
A job starts when its dependencies finished OK, its wait_file exists, and enough GPUs have (capacity - reserved) >= mem
and nvidia-smi free >= mem. CUDA_VISIBLE_DEVICES is set by the scheduler.
Each job writes its exit code to logs/job_<id>.rc, so the scheduler can be restarted at any time: running jobs are
re-adopted from the state file. To retry a failed job: touch logs/job_<id>.retry.
State: logs/sched_<host>.state.json, job logs: logs/job_<id>.log.
"""
import json, os, socket, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOST = socket.gethostname(); CAP = float(os.environ.get("SCHED_CAP_GB", "46"))
JOBS = os.path.join(ROOT, "logs", f"jobs_{HOST}.jsonl"); STATE = os.path.join(ROOT, "logs", f"sched_{HOST}.state.json")
ENV = os.path.join(ROOT, "logs", f"env_{HOST}.sh")
L = lambda jid, ext: os.path.join(ROOT, "logs", f"job_{jid}.{ext}")


def smi_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return {int(a): int(b) / 1024 for a, b in (l.split(", ") for l in out.strip().splitlines())}


def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    running = {k: (None, v["gpus"], v.get("mem", 0)) for k, v in st.items() if v and v.get("status") == "running"}
    for k in running:
        del st[k]
    while True:
        jobs = [json.loads(l) for l in open(JOBS) if l.strip()] if os.path.exists(JOBS) else []
        for j in jobs:
            if os.path.exists(L(j["id"], "retry")) and j["id"] not in running:
                st.pop(j["id"], None); os.remove(L(j["id"], "retry"))
                for k, v in list(st.items()):               # dependants skipped because of it get another chance
                    if v and v.get("status", "").startswith("skipped"):
                        del st[k]
        for jid, (p, gpus, mem) in list(running.items()):
            if p is not None:
                p.poll()
            if os.path.exists(L(jid, "rc")):
                rc = int(open(L(jid, "rc")).read().strip() or 1)
                st[jid] = {"status": "ok" if rc == 0 else f"fail({rc})", "end": time.strftime("%F %T"), "gpus": gpus}
                del running[jid]; print(time.strftime("%T"), jid, st[jid]["status"], flush=True)
        reserved = {}
        for _, gpus, mem in running.values():
            for g in gpus:
                reserved[g] = reserved.get(g, 0) + mem
        free = smi_free()
        for j in jobs:
            if j["id"] in st or j["id"] in running:
                continue
            deps = [st.get(d, {}).get("status") for d in j.get("after", [])]
            if any(s is not None and s != "ok" for s in deps):
                st[j["id"]] = {"status": "skipped(dep failed)"}; continue
            if not all(s == "ok" for s in deps):
                continue
            if j.get("wait_file") and not os.path.exists(os.path.join(ROOT, j["wait_file"])):
                continue
            ok = [g for g in sorted(free, key=lambda g: reserved.get(g, 0)) if CAP - reserved.get(g, 0) >= j["mem"] and free[g] >= j["mem"]]
            if len(ok) < j.get("gpus", 1):
                continue
            gpus = sorted(ok[:j.get("gpus", 1)])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
            for ext in ("rc",):
                if os.path.exists(L(j["id"], ext)):
                    os.remove(L(j["id"], ext))
            log = open(L(j["id"], "log"), "w")
            p = subprocess.Popen(["bash", "-c", f". {ENV}; {j['cmd']}; echo $? > {L(j['id'], 'rc')}"], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            running[j["id"]] = (p, gpus, j["mem"])
            for g in gpus:
                reserved[g] = reserved.get(g, 0) + j["mem"]; free[g] -= j["mem"]
            print(time.strftime("%T"), "start", j["id"], "gpus", gpus, flush=True)
        json.dump({**st, **{k: {"status": "running", "gpus": v[1], "mem": v[2]} for k, v in running.items()}}, open(STATE, "w"), indent=1)
        time.sleep(20)


if __name__ == "__main__":
    main()
