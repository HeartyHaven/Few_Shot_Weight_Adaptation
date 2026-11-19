"""Build the data splits.

  data/<task>/dev.jsonl        visible to the teacher (100 items, with answers)
  _secret/<task>/val.jsonl     hidden; the teacher only ever sees aggregate scores (max 10 looks)
  _secret/<task>/test_*.jsonl  hidden; evaluator only

Usage: python -m tdwa.data
"""
import json, os, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEOPLE = ["Alice", "Bob", "Claire", "Dave", "Eve", "Fred", "Gertrude", "Hank", "Iris"]

THEMES = {
    "books": dict(
        intro="{names} are friends and avid readers who occasionally trade books. At the start of the semester, they each buy one new book: {init}.",
        each="{p} gets {o}", mid="As the semester proceeds, they start trading around the new books. {swaps} At the end of the semester, {q} has",
        swap="{a} and {b} swap books.",
        pool=["Dune", "Emma", "Walden", "Beloved", "Hamlet", "Ivanhoe", "Persuasion", "Middlemarch", "The Hobbit", "Dracula",
              "Jane Eyre", "The Trial", "Candide", "Siddhartha"], opt="{o}"),
    "balls": dict(
        intro="{names} are playing a game. At the start of the game, they are each holding a ball: {init}.",
        each="{p} has a {o} ball", mid="As the game progresses, pairs of players trade balls. {swaps} At the end of the game, {q} has the",
        swap="{a} and {b} swap balls.",
        pool=["red", "blue", "green", "yellow", "orange", "pink", "brown", "white", "black", "purple", "grey", "teal", "gold", "silver"],
        opt="{o} ball"),
    "dance": dict(
        intro="{names} are dancers at a square dance. At the start of a song, they each have a partner: {init}.",
        each="{p} is dancing with {o}", mid="Throughout the song, the dancers often trade partners. {swaps} At the end of the dance, {q} is dancing with",
        swap="{a} and {b} switch partners.",
        pool=["Karl", "Lena", "Marco", "Nadia", "Oscar", "Priya", "Quinn", "Rosa", "Stefan", "Tara", "Umar", "Vera", "Wade", "Yuki"],
        opt="{o}"),
    "gifts": dict(
        intro="{names} are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color: {init}.",
        each="{p} has a {o} present", mid="As the event progresses, pairs of people swap gifts. {swaps} At the end of the event, {q} has the",
        swap="{a} and {b} swap their gifts.",
        pool=["red", "blue", "green", "yellow", "orange", "pink", "brown", "white", "black", "purple", "grey", "teal", "gold", "silver"],
        opt="{o} present"),
    "soccer": dict(
        intro="{names} are on the same team in a soccer match. At the start of the match, they are each assigned to a position: {init}.",
        each="{p} is playing {o}", mid="As the game progresses, pairs of players occasionally swap positions. {swaps} At the end of the match, {q} is playing",
        swap="{a} and {b} trade positions.",
        pool=["goalkeeper", "striker", "left winger", "right winger", "left midfielder", "right midfielder", "center midfielder",
              "left back", "right back", "sweeper", "fullback", "cheerleader", "benchwarmer"], opt="{o}"),
}


def join(xs):
    return ", ".join(xs[:-1]) + ", and " + xs[-1]


def make_item(rng, n, n_swaps=None, uid=""):
    """One tracking item in the BBH format. Returns prompt text, gold option, and the full ground-truth trace."""
    n_swaps = n_swaps or n
    th_name = rng.choice(sorted(THEMES)); th = THEMES[th_name]
    people = PEOPLE[:n]
    objs = rng.sample(th["pool"], n)
    state = dict(zip(people, objs))
    init = join([th["each"].format(p=p, o=o) for p, o in state.items()])
    swaps, trace = [], [dict(state)]
    for _ in range(n_swaps):
        a, b = rng.sample(people, 2)
        state[a], state[b] = state[b], state[a]
        swaps.append((a, b)); trace.append(dict(state))
    lead = ["First,"] + ["Then,"] * (n_swaps - 2) + ["Finally,"] if n_swaps > 1 else ["First,"]
    swap_txt = " ".join(f"{w} {th['swap'].format(a=a, b=b)}" for w, (a, b) in zip(lead, swaps))
    q = rng.choice(people)
    opts = [th["opt"].format(o=o) for o in objs]
    text = (th["intro"].format(names=join(people), init=init) + "\n" + th["mid"].format(swaps=swap_txt, q=q) +
            "\nOptions:\n" + "\n".join(f"({'ABCDEFGHI'[i]}) {o}" for i, o in enumerate(opts)))
    gold = "(" + "ABCDEFGHI"[objs.index(state[q])] + ")"
    return {"id": uid, "task": "t2", "group": f"{n}obj", "n_objects": n, "n_swaps": n_swaps, "theme": th_name,
            "question": text, "answer": gold, "swaps": swaps, "asked": q, "trace": trace}


def build_t2():
    def split(seed, counts, prefix, swaps_fn=None):
        rng = random.Random(seed); out = []
        for n, c in counts:
            for i in range(c):
                out.append(make_item(rng, n, swaps_fn(n) if swaps_fn else None, f"{prefix}-{n}-{i}"))
        return out
    dev = split(1101, [(3, 20), (5, 40), (7, 40)], "dev")
    val = split(2202, [(3, 30), (5, 60), (7, 60)], "val")
    test = split(3303, [(3, 100), (5, 100), (7, 100)], "fresh")
    ood = split(4404, [(5, 50), (7, 50), (9, 50)], "ood", swaps_fn=lambda n: n + 4 if n < 9 else n)
    from datasets import load_dataset
    bbh = []
    for k, n in (("three", 3), ("five", 5), ("seven", 7)):
        for i, r in enumerate(load_dataset("lukaemon/bbh", f"tracking_shuffled_objects_{k}_objects")["test"]):
            bbh.append({"id": f"bbh-{n}-{i}", "task": "t2", "group": f"{n}obj", "n_objects": n, "question": r["input"], "answer": r["target"]})
    return {"dev": dev}, {"val": val, "test_fresh": test, "test_ood": ood, "test_bbh": bbh}


# Calculators the teacher is asked to teach (T1). The other 11 rule-based calculators are never taught and
# serve as the near-domain control. Chosen before any teaching run to span easy to hard (see pilot/REPORT.md).
T1_TEACH = [45, 51, 48, 8, 20, 15, 25, 4]   # CURB-65, SIRS, PERC, Wells PE, Centor, Child-Pugh, HAS-BLED, CHA2DS2-VASc
RULE_CATS = ("risk", "severity", "diagnosis")


def build_t1():
    from datasets import load_dataset
    mc = load_dataset("nsk7153/MedCalc-Bench-Verified")
    def conv(r, pre):
        return {"id": f"{pre}-{r['Row Number']}", "task": "t1", "group": r["Calculator Name"], "calc_id": r["Calculator ID"],
                "question": f"Patient note:\n{r['Patient Note']}\n\nQuestion: {r['Question']}",
                "answer": r["Ground Truth Answer"], "lower": float(r["Lower Limit"]), "upper": float(r["Upper Limit"]),
                "explanation": r["Ground Truth Explanation"]}
    rng = random.Random(5505)
    by = {}
    for r in mc["train"]:
        if r["Calculator ID"] in T1_TEACH:
            by.setdefault(r["Calculator ID"], []).append(r)
    dev, val = [], []
    for cid in T1_TEACH:
        rows = by[cid]; rng.shuffle(rows)
        k = 13 if cid in T1_TEACH[:4] else 12            # 4*13 + 4*12 = 100
        dev += [conv(r, "dev") for r in rows[:k]]
        val += [conv(r, "val") for r in rows[k:k + 15]]
    test = [conv(r, "test") for r in mc["test"] if r["Category"] in RULE_CATS]
    for x in test:
        x["taught"] = x["calc_id"] in T1_TEACH
    for x in val:
        x.pop("explanation")
    return {"dev": dev}, {"val": val, "test": test}


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_canary():
    """Regression guard used after every edit: 60 ARC-Easy multiple-choice + 40 two-digit arithmetic items."""
    from datasets import load_dataset
    rng = random.Random(7707); out = []
    arc = [r for r in load_dataset("allenai/ai2_arc", "ARC-Easy")["validation"] if r["choices"]["label"][0] == "A" and len(r["choices"]["label"]) <= 5]
    for r in rng.sample(arc, 60):
        opts = "\n".join(f"({l}) {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
        out.append({"kind": "mc", "prompt": f"{r['question']}\n{opts}\nAnswer with the option letter in parentheses.",
                    "answer": r["answerKey"], "n": len(r["choices"]["label"])})
    for _ in range(40):
        a, b = rng.randint(11, 99), rng.randint(11, 99); op = rng.choice("+-")
        out.append({"kind": "arith", "prompt": f"What is {a} {op} {b}? Answer with the number only.", "answer": str(a + b if op == "+" else a - b)})
    return out


if __name__ == "__main__":
    import shutil
    write(f"{ROOT}/data/canary.jsonl", build_canary())
    shutil.copy("/mnt/data2/users/lang.gao/Editable_Real/generic.txt", f"{ROOT}/data/generic.txt")
    for task, fn in (("t2", build_t2), ("t1", build_t1)):
        pub, sec = fn()
        for k, v in pub.items():
            write(f"{ROOT}/data/{task}/{k}.jsonl", v); print(task, k, len(v))
        for k, v in sec.items():
            write(f"{ROOT}/_secret/{task}/{k}.jsonl", v); print(task, k, len(v), "(secret)")
