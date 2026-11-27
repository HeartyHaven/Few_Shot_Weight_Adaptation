"""Prompt formats and graders shared by the teaching sandbox and the evaluator."""
import re

MODE = {
    "t1": {"free": "Work step by step, then finish with a line 'Final answer: <number>'.",
           "direct": "Answer with only the final number, nothing else."},
    "t2": {"free": "Work step by step, then finish with a line 'Final answer: (X)'.",
           "direct": "Answer with only the option letter in parentheses, for example (A)."},
}
MODE["t3"] = MODE["t2"]                 # DDXPlus: same multiple-choice answer format (5 options)
OPT_TASKS = ("t2", "t3")
NUM = re.compile(r"-?\d+(?:\.\d+)?")
OPT = re.compile(r"\(([A-I])\)")


def format_prompt(item, mode="free"):
    """The exact user message used in every quiz, validation and final test. No procedure text is ever added."""
    return item["question"] + "\n\n" + MODE[item["task"]][mode]


def parse(task, text, mode="free"):
    free = mode == "free"
    if free and "final answer" in text.lower():
        tail = text[text.lower().rindex("final answer"):]
        if task in OPT_TASKS:
            m = OPT.search(tail) or re.search(r"\b([A-I])\b", tail[len("final answer"):])
            return f"({m.group(1)})" if m else None
        m = NUM.search(tail.replace(",", ""))
        return float(m.group()) if m else None
    if task in OPT_TASKS:
        ms = OPT.findall(text)
        if not ms:
            m = re.match(r"\s*([A-I])\b", text)
            return f"({m.group(1)})" if m else None
        return f"({ms[-1] if free else ms[0]})"
    ms = NUM.findall(text.replace(",", ""))
    return float(ms[-1] if free else ms[0]) if ms else None


def is_correct(item, pred):
    if pred is None:
        return False
    if item["task"] in OPT_TASKS:
        return pred == item["answer"]
    lo = item.get("lower", float(item["answer"])); hi = item.get("upper", float(item["answer"]))
    return lo <= pred <= hi


def grade(item, output, mode="free"):
    pred = parse(item["task"], output, mode)
    return {"pred": pred, "correct": bool(is_correct(item, pred))}


# ---- T4 (MATH Level 5) and T5 (CRUXEval-O): answers are strings, compared after normalisation (2026-11-22) ----
import ast as _ast
from fractions import Fraction as _Fr

MODE["t4"] = {"free": "Work step by step, then finish with a line 'Final answer: <answer>'.",
              "direct": "Answer with only the final answer, nothing else."}
MODE["t5"] = {"free": "Work step by step, then finish with a line 'Final answer: <the returned value as a Python literal>'.",
              "direct": "Answer with only the returned value as a Python literal, nothing else."}
STR_TASKS = ("t4", "t5")


def _boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    d = 0
    for k in range(j, len(s)):
        d += (s[k] == "{") - (s[k] == "}")
        if d == 0:
            return s[j + 1:k]
    return None


def _tail_line(text):
    """The answer after the last 'Final answer': a \\boxed{} there if any, else its first non-empty line that is not only
    a math/code fence ($$, \\[, ```); markdown bold and <answer> tags removed. Without 'Final answer': the last \\boxed{}."""
    lo = text.lower()
    if "final answer" in lo:
        t = text[lo.rindex("final answer") + len("final answer"):]
        b = _boxed(t)
        if b is not None and b.strip():
            return b.strip()
        for line in t.split("\n"):
            x = re.sub(r"</?answer>", "", line).strip().lstrip(":* ").strip()      # '**Final answer:** x', ': **x**'
            while x.endswith("**"):
                x = x[:-2].rstrip()
            if len(x) > 6 and x.startswith("```") and x.endswith("```"):
                x = x.strip("`").removeprefix("python").strip()
            if not x or x in ("$$", "\\[", "\\]", "$") or x.startswith("```"):
                continue
            return x
        return None
    b = _boxed(text)
    return b.strip() if b else None


def _math_norm(s):
    s = s.strip()
    while s and s[-1] in ".$ ":
        s = s[:-1]
    s = s.lstrip("$ ")
    b = _boxed(s)
    if b is not None:
        s = b
    for a, c in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""), ("\\!", ""), ("{,}", ""), ("\\,", ""),
                 ("\\$", ""), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", "")):
        s = s.replace(a, c)
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s); s = re.sub(r"\\mbox\{(.*?)\}", r"\1", s)
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)              # 'n = 255'
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)   # \frac19
    s = s.replace(" ", "").rstrip(".")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    s = re.sub(r"_\{(\w+)\}", r"_\1", s)
    return s


def _math_value(s):
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m:
        return _Fr(int(m.group(2)), int(m.group(3))) * (-1 if m.group(1) else 1)
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        return _Fr(int(m.group(1)), int(m.group(2)))
    try:
        return _Fr(s) if re.fullmatch(r"-?\d+(\.\d+)?", s) else None
    except Exception:
        return None


def _py_value(s):
    s = s.strip()
    if s.startswith("`") and s.endswith("`"):
        s = s.strip("`").strip()
    if "==" in s and s.lstrip().startswith(("f(", "assert")):
        s = s.split("==", 1)[1].strip()
    try:
        return ("lit", _ast.literal_eval(s))
    except Exception:
        return ("str", s)


def _parse_str(task, text):
    return _tail_line(text)


def _correct_str(item, pred):
    if pred is None:
        return False
    if item["task"] == "t4":
        p, g = _math_norm(pred), _math_norm(item["answer"])
        if p == g:
            return True
        vp, vg = _math_value(p), _math_value(g)
        return vp is not None and vp == vg
    vp, vg = _py_value(pred), _py_value(item["answer"])
    return vp == vg and (vp[0] == "lit" or vp[1] == vg[1])


_parse_num, _is_correct_num = parse, is_correct


def parse(task, text, mode="free"):
    if task in STR_TASKS:
        if mode != "free":
            return text.strip().split("\n")[0].strip() or None
        return _parse_str(task, text)
    return _parse_num(task, text, mode)


def is_correct(item, pred):
    if item["task"] in STR_TASKS:
        return _correct_str(item, pred)
    return _is_correct_num(item, pred)
