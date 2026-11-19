"""Shared task board for everyone (people, agents, other sessions) working on this project.

The board lives in the GitHub repository: tasks/board.json is the source of truth, TASKS.md is rendered from it.
Every change is committed and pushed immediately, and a push that loses a race is retried after re-reading the
board, so two workers can never both claim the same task.

  python -m tdwa.board list [--status open]          show tasks (pulls first)
  python -m tdwa.board claim  ID --owner NAME        take an open task (fails if someone else has it)
  python -m tdwa.board done   ID --owner NAME --result "..." [--output PATH]
  python -m tdwa.board fail   ID --owner NAME --result "why"
  python -m tdwa.board release ID --owner NAME --note "why"   give a claimed task back
  python -m tdwa.board note   ID --owner NAME --note "progress text"
  python -m tdwa.board render                         rewrite TASKS.md locally (no push)
Add --no-push to work offline (then push yourself later).
"""
import argparse, datetime, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = os.path.join(ROOT, "tasks", "board.json")
MD = os.path.join(ROOT, "TASKS.md")
STATUS_ZH = {"open": "未开始（可认领）", "claimed": "已认领", "running": "进行中", "done": "已完成", "failed": "失败",
             "void": "作废", "blocked": "受阻"}
ORDER = ["running", "claimed", "blocked", "open", "done", "failed", "void"]


def now():
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")


def git(*a, check=True):
    r = subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(a)} failed: {r.stderr.strip()}")
    return r


def pull():
    git("pull", "--rebase", "--autostash", "-q", check=False)


def load():
    return json.load(open(BOARD))


def save(b):
    b["updated"] = now()
    json.dump(b, open(BOARD, "w"), ensure_ascii=False, indent=1); render(b)


def cell(x):
    return str(x or "").replace("|", "\\|").replace("\n", " ")


def render(b=None):
    b = b or load()
    out = ["# 任务看板", "",
           f"> 由 `python -m tdwa.board render` 从 `tasks/board.json` 生成，请不要手改本文件。最后更新：{b.get('updated', '')}。",
           "> 认领和完成任务的规则见 [README.md](README.md#多人并行协作规则)。", ""]
    for st in ORDER:
        ts = [t for t in b["tasks"] if t["status"] == st]
        if not ts:
            continue
        out += [f"## {STATUS_ZH[st]}（{len(ts)}）", "",
                "| ID | 任务 | 学生 | 教师 | 数据 | 负责者 | 时间 | 结果 / 备注 | 产出位置 |", "|---|---|---|---|---|---|---|---|---|"]
        for t in ts:
            when = t.get("finished_at") or t.get("claimed_at") or ""
            out.append("| " + " | ".join(cell(x) for x in (t["id"], t["title"], t.get("student"), t.get("teacher"), t.get("dataset"), t.get("owner"),
                                                                 when, t.get("result") or t.get("note"), t.get("output"))) + " |")
        out.append("")
    open(MD, "w").write("\n".join(out) + "\n")


def find(b, tid):
    for t in b["tasks"]:
        if t["id"] == tid:
            return t
    sys.exit(f"no task {tid}")


def change(tid, fn, msg, push=True, tries=5):
    """Pull, apply fn(task) (which may raise SystemExit to refuse), commit, push; retry if the push loses a race."""
    for i in range(tries):
        if push:
            pull()
        b = load(); t = find(b, tid); fn(t); save(b)
        if not push:
            return t
        git("add", BOARD, MD); git("commit", "-q", "-m", f"board: {msg}", check=False)
        r = git("push", "-q", check=False)
        if r.returncode == 0:
            return t
        git("reset", "-q", "--hard", "HEAD~1"); time.sleep(2 + 3 * i)          # lost the race: re-read and re-check
    sys.exit("could not push the board after several tries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "claim", "done", "fail", "release", "note", "render", "start"])
    ap.add_argument("id", nargs="?"); ap.add_argument("--owner"); ap.add_argument("--result"); ap.add_argument("--output")
    ap.add_argument("--note"); ap.add_argument("--status"); ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args(); push = not a.no_push
    if a.cmd == "render":
        render(); return
    if a.cmd == "list":
        if push:
            pull()
        for t in load()["tasks"]:
            if not a.status or t["status"] == a.status:
                print(f"{t['id']:28s} {t['status']:8s} {t.get('owner') or '-':22s} {t['title']}")
        return
    if not a.id or not a.owner:
        sys.exit("need a task id and --owner")

    def claim(t):
        if t["status"] != "open":
            sys.exit(f"{t['id']} is {t['status']} (owner {t.get('owner')}); pick another task")
        t.update(status="claimed", owner=a.owner, claimed_at=now(), note=a.note or t.get("note"))

    def start(t):
        if t.get("owner") not in (None, a.owner) and t["status"] in ("claimed", "running"):
            sys.exit(f"{t['id']} belongs to {t['owner']}")
        t.update(status="running", owner=a.owner, claimed_at=t.get("claimed_at") or now(), note=a.note or t.get("note"))

    def finish(st):
        def f(t):
            if t.get("owner") not in (None, a.owner):
                sys.exit(f"{t['id']} belongs to {t['owner']}")
            t.update(status=st, owner=a.owner, finished_at=now(), result=a.result or t.get("result"), output=a.output or t.get("output"))
        return f

    def release(t):
        if t.get("owner") != a.owner:
            sys.exit(f"{t['id']} belongs to {t.get('owner')}")
        t.update(status="open", owner=None, claimed_at=None, note=a.note)

    def note(t):
        t["note"] = a.note

    fn = {"claim": claim, "start": start, "done": finish("done"), "fail": finish("failed"), "release": release, "note": note}[a.cmd]
    t = change(a.id, fn, f"{a.cmd} {a.id} ({a.owner})", push)
    print(f"{t['id']}: {t['status']} by {t.get('owner')}")


if __name__ == "__main__":
    main()
