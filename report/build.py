"""Merge exported run data with Chinese captions/translations into the files the report page fetches (report/site/)."""
import json, os
D = os.path.dirname(os.path.abspath(__file__)); os.makedirs(D + "/site", exist_ok=True)
def rd(p):
    return open(p).read() if os.path.exists(p) else ""
EDIT_ZH = {
 "fable_s0": {
  1: {"what": "让负责查表的三个注意力头对准正确的行", "why": "学生知道规则但执行不好：512 步里 12.5% 写错；过程全对的答卷里仍有 34% 最后读错行。第 17 层 0 号头、第 14 层 2 号头、第 13 层 2 号头在答对时把 0.46 的注意力放在正确行上，答错时只有 0.09。线性探针显示学生内部其实知道该读哪一行。",
      "how": "只改这三个头的查询权重（q_proj）中对位置不敏感的那部分行。对每个需要查表的位置，先算出“能让注意力落到正确那一行”的目标向量，再用带正则的最小二乘公式一次求出权重增量；同时要求在其他任务位置和 111 条普通文本上增量的输出为零。强度系数用真题上的逐点检查挑选。没有梯度下降。",
      "pred": "真题自由作答从 0.54 升到 0.72–0.76（未提交时实测 0.74）；步骤执行错误从 67 降到 40–45；通用困惑度变化在 3% 以内，选择题检查不变，算术检查最多丢 2 道。", "result": "预测成立。隐藏验证 67.3%。保留。"},
  2: {"what": "让学生按题目顺序执行每一次交换", "why": "长题（7 人 10–14 次交换、9 人）里只有 36/76 份答卷的步骤顺序完全正确：遇到重复出现的一对人会往回跳，或者循环。负责抄步骤标题的是第 17 层 0 号头和第 11 层 1 号头。",
      "how": "和编辑 1 同样的最小二乘方法，只作用在第 11 层 1 号头的查询权重上（98 行）。目标是让它在写每个步骤标题时对准题目里下一次交换的人名。加了约束：在第一张表之前和最后一次交换之后不触发。",
      "pred": "长题步骤顺序全对的从 36/76 升到约 48/76；普通真题正确率变化不超过 ±3 个百分点（实测 0.74 → 0.76）；长题正确率只会略升，因为交换执行错误仍占主导。", "result": "预测成立。隐藏验证 68.7%。保留。"},
  3: {"what": "修多词物品名被写乱的问题", "why": "编辑 1 之后，交换行的第一个词读自对方的行，后续的词却回到自己原来的行，出现“Middle Siddhartha”这类拼接。真题自由作答里有 21 行被写乱。",
      "how": "和编辑 1 同样的方法、同样的三个头，目标换成物品名的后续词位置。强度 20。",
      "pred": "子技能指标：后续词准确率 0.975 → 约 0.99，写乱的行 21 → 10 以下。整题正确率预计只有 0–3 个百分点的小提升，所以事先约定：隐藏验证不低于 0.66 才保留，否则撤回。", "result": "子技能预测成立（0.993，写乱的行降到 6），但隐藏验证掉到 62.7%，低于约定门槛。已撤回。"}},
 "astra_s0": {
  1: {"what": "把两个帮倒忙的注意力头输出减半", "why": "在 12 道真题上逐个关掉注意力头：关掉第 13 层 1 号头后状态损失从 0.242 降到 0.087；关掉第 16 层 1 号头后读结果损失从 9.10 降到 5.90。两个头都减半时，30 道题的正确率从 50% 到 60%。",
      "how": "把第 13 层和第 16 层输出权重（o_proj）的第 256–511 列乘以 0.5，等于把这两层的 1 号头贡献减半。从 5 个候选倍数里挑选。没有梯度。", "pred": "真题正确率提高约 5–10 个百分点；困惑度和检查题在限度内。", "result": "隐藏验证没有变化（53.3% → 53.3%）。保留。"},
  2: {"what": "把负责抄写状态的头放大 1.5 倍", "why": "第 17 层 0 号头在结构上能找到正确的上一个物品（平均 0.751 的注意力，96.4% 首选正确），但它的贡献会被别的部件盖过。",
      "how": "把第 17 层输出权重的前 256 列（0 号头）乘以 1.5。从 1.5、2、3 三个倍数和两种“让注意力更尖”的试验里挑选。", "pred": "真题至少提高 5 个百分点；隐藏验证提高 3–7 个百分点。", "result": "隐藏验证 57.3%（+4 个百分点）。保留。"},
  3: {"what": "修正选项字母的查找", "why": "即使状态和口头答案都对，第 17 层 0 号头有时会看向错误的选项字母。",
      "how": "对第 17 层查询权重的前 256 行做闭式岭回归修正，用 30 道真题上的 60 个选项位置拟合。三个事先定好的正则强度里选了 0.01。", "pred": "选项探针 70% → 100%；其余真题提高 5–10 个百分点；隐藏验证提高 3–6 个百分点。", "result": "拟合用的 30 道题多对 3 道，其余 70 道少对 5 道；隐藏验证 56.7%，低于编辑 2。教师判断为过拟合，已撤回。"}},
 "astra_s1": {
  1: {"what": "把一个干扰物品绑定的头调到 0.75 倍", "why": "自由作答时状态转移的错误会累积。关掉第 16 层 1 号头能降低“发生变化的物品”的损失。",
      "how": "把第 16 层输出权重的第 256–511 列乘以 0.75。倍数从 0.5、0.75、1、1.25、1.5 的观察里选一个。定位阶段逐个关掉了 104 个注意力头。", "pred": "真题 54% → 约 57%；隐藏验证 53.33% → 55–58%；困惑度上升小于 3%。", "result": "真题 57%，隐藏验证 54.0%（150 题多对 1 道），低于预测。保留。"}}}
for key in sorted(x[:-5] for x in os.listdir(D + "/data") if x.endswith(".json") and not x.endswith("_quizzes.json")):
    d = json.load(open(f"{D}/data/{key}.json"))
    cap = {}
    if os.path.exists(f"{D}/captions/{key}.json"):
        cap = {c["n"]: c for c in json.load(open(f"{D}/captions/{key}.json"))}
    for s in d["steps"]:
        s["cap"] = cap.get(s["n"])
    d["syllabus_zh"] = rd(f"{D}/captions/{key}_syllabus_zh.md"); d["log_zh"] = rd(f"{D}/captions/{key}_log_zh.md")
    removed = {e for r in d["rollbacks"] for e in r["removed"]}
    for e in d["edits"]:
        e["zh"] = EDIT_ZH.get(key, {}).get(e["edit_id"]); e["rolled_back"] = e["edit_id"] in removed
        e["data_used"] = e["data_used"] if isinstance(e["data_used"], str) else json.dumps(e["data_used"], ensure_ascii=False)
    c = d["config"]; d["config"] = {k: c.get(k) for k in ("student", "teacher", "seed", "fte_budget", "mem_ratio", "token_budget", "max_turns", "val_looks", "wall_seconds", "turns", "validation_looks", "scaffold")}
    json.dump(d, open(f"{D}/site/{key}.json", "w"), ensure_ascii=False)
    q = json.load(open(f"{D}/data/{key}_quizzes.json"))
    for x in q:
        for it in x["items"]:
            if len(it["q"]) > 6000: it["q"] = it["q"][:6000] + " …[截断]"
            if len(it["out"]) > 6000: it["out"] = it["out"][:6000] + " …[截断]"
    json.dump(q, open(f"{D}/site/{key}_quizzes.json", "w"), ensure_ascii=False)
    print(key, "captions", len(cap), "zh syllabus", len(d["syllabus_zh"]), "zh log", len(d["log_zh"]))
