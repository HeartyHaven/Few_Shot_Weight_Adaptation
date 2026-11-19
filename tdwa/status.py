"""Print a one-screen status of all teaching runs: python -m tdwa.status"""
import glob, json, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for R in sorted(glob.glob(os.path.join(ROOT, "runs", "*"))):
    cfg = json.load(open(R + "/config.json"))
    led = [json.loads(l) for l in open(R + "/ledger.jsonl")] if os.path.exists(R + "/ledger.jsonl") else []
    ev = [json.loads(l) for l in open(R + "/transcript.jsonl")]
    tools = [e for e in ev if e["event"] == "tool"]
    last = time.time() - ev[-1]["t"] if ev else -1
    print(f"== {os.path.basename(R)}  {'FINISHED' if cfg.get('finished') else 'running'}  tool_calls={len(tools)}  last_event={last/60:.1f} min ago")
    print(f"   quizzes={len(os.listdir(R+'/quizzes'))} edits={len(os.listdir(R+'/edits'))} fwd_tokens={sum(l.get('fwd_tokens',0) for l in led):,} "
          f"bwd_tokens={sum(l.get('bwd_tokens',0) for l in led):,} max_mem_ratio={max([l.get('peak_mem_ratio',0) for l in led] or [0])}")
    a = [e for e in ev if e["event"] == "assistant"]
    if a: print("   teacher tokens:", a[-1]["cumulative"])
    for l in led:
        if l["event"] == "commit": print(f"   edit {l['edit']}: accepted={l['accepted']} values={l['n_values']} reg={l['regression']}")
        if l["event"] == "rollback": print(f"   rollback to {l['to']}: {l['reason'][:100]}")
    for f in sorted(glob.glob(R + "/exams/*.json")):
        x = json.load(open(f)); print(f"   exam {x['look']} (after edit {x['after_edit']}): {x['accuracy']} {x['by_group']}")
