"""Prompt formats and graders shared by the teaching sandbox and the evaluator."""
import re

MODE = {
    "t1": {"free": "Work step by step, then finish with a line 'Final answer: <number>'.",
           "direct": "Answer with only the final number, nothing else."},
    "t2": {"free": "Work step by step, then finish with a line 'Final answer: (X)'.",
           "direct": "Answer with only the option letter in parentheses, for example (A)."},
}
NUM = re.compile(r"-?\d+(?:\.\d+)?")
OPT = re.compile(r"\(([A-I])\)")


def format_prompt(item, mode="free"):
    """The exact user message used in every quiz, validation and final test. No procedure text is ever added."""
    return item["question"] + "\n\n" + MODE[item["task"]][mode]


def parse(task, text, mode="free"):
    free = mode == "free"
    if free and "final answer" in text.lower():
        tail = text[text.lower().rindex("final answer"):]
        if task == "t2":
            m = OPT.search(tail) or re.search(r"\b([A-I])\b", tail[len("final answer"):])
            return f"({m.group(1)})" if m else None
        m = NUM.search(tail.replace(",", ""))
        return float(m.group()) if m else None
    if task == "t2":
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
    if item["task"] == "t2":
        return pred == item["answer"]
    lo = item.get("lower", float(item["answer"])); hi = item.get("upper", float(item["answer"]))
    return lo <= pred <= hi


def grade(item, output, mode="free"):
    pred = parse(item["task"], output, mode)
    return {"pred": pred, "correct": bool(is_correct(item, pred))}
