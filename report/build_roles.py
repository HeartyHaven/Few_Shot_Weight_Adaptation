"""Merge report/captions/roles/<run>.json (component roles extracted from the teaching logs) into report/site/atlas_roles.json."""
import glob, json, os
D = os.path.dirname(os.path.abspath(__file__))
out = {}
for p in sorted(glob.glob(D + "/captions/roles/*.json")):
    d = json.load(open(p)); out[d.get("run") or os.path.basename(p)[:-5]] = {"summary_zh": d.get("summary_zh", ""), "items": d.get("items", [])}
json.dump(out, open(D + "/site/atlas_roles.json", "w"), ensure_ascii=False)
print(len(out), "runs,", sum(len(v["items"]) for v in out.values()), "items")
