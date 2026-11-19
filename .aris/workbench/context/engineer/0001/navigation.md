# ARIS navigation for every role

Workbench host: **herdr**. Recover its original identity from launch.json; do not infer the host from the agent harness.

ARIS provides research workflows, project memory and evidence-based acceptance. Use a **pipeline** for an authorized complete research lifecycle, a **workflow** for one connected stage, or an individual **skill** for a bounded task. The main owner can delegate any of those granularities; opening this navigation starts none of them.

First read `/mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation/.aris/workbench/context/engineer/0001/role.md` for your responsibility, scoped task and write boundaries, and `/mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation/.aris/workbench/context/engineer/0001/recovery.md` for the project goal/current-state originals. Read the exact required SKILL.md and referenced resources before executing its workflow.

## Stages and capabilities

- **文献检索**: arXiv/OpenAlex/Semantic Scholar 等文献搜索、抓取与阅读
- **选题与方向**: 找 idea、查新、方向打磨、research wiki 与总编排 pipeline
- **自动评审闭环**: 跨模型自动 review 循环与对抗性评审
- **公式与证明**: 公式推导、证明撰写与证明审查
- **实验与算力**: 实验计划/执行/监控/分析与 GPU 集群、无服务器算力
- **论文写作主线**: 从大纲到 PDF 的写作、编译、数字/引用/诚信审计
- **配图与展示**: 插图、示意图、海报、幻灯片与演讲
- **投稿与基金**: 返修 rebuttal、重投 pipeline、基金申请书
- **专利**: 发明构建、权利要求、实施例到各国格式的专利流水线
- **元工具与杂项**: skill 语料自优化、通知、面试准备等

The local catalog `/mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation/.aris/workbench/context/engineer/0001/catalog/SKILLS_CATALOG.md` describes skills and external dependencies. The verbatim canonical selection/dependency catalog is `/mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation/.aris/workbench/context/engineer/0001/catalog/skill-groups.tsv` (source `/home/lang.gao/.cache/superaris/aris/2901234ad2954cd2388ced8d3aed2ed9de42803a/tools/skill-groups.tsv`). These are navigation and descriptions, not all skill bodies.

## Skills prepared for this role

- project-agent: `/mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation/.aris/workbench/context/engineer/0001/skills/project-agent/SKILL.md`

When a workflow invokes a named child skill, use this exact index; native slash/menu discovery is optional. Only the requested skills and default dependency closure have been copied.

## On-demand skills

Choose the skill by name from the local catalog. The command-capable project owner can prepare it through the existing source resolver, host customization precedence and catalog dependency closure:

```sh
/home/lang.gao/.local/pipx/venvs/superaris/bin/python -m superaris context /mnt/data2/users/lang.gao/Few_Shot_Weight_Adaptation --role engineer --skill SKILL_NAME
```

Replace SKILL_NAME with the chosen catalog name. This preparation preserves the role's current task, recovery and scopes by default and returns a new packet; it does not send native input or execute the skill. Read the returned packet/index and then the actual skill before doing authorized work. If your read domain or file-only tools cannot execute/access this repository command, hand the named skill request to the main owner through the existing task channel and read the prepared project-local packet when returned. Users need not reconstruct command parameters or install another skill set.

## Capabilities and tools

Catalog entries describe possible tools, not verified accounts, MCP services, GPUs or reviewer availability. Check the selected skill's actual requirements and use the existing owner for unsupported execution. Set the packet's ARIS_REPO explicitly in the same helper invocation. Preserve original MCP routes, reviewer freshness, budgets, paused Goals and publication/merge permissions. Local files or submitted messages do not prove native loading, actual reading or permanent model memory. After compact/restart, re-read recovery, the current role packet and its required skills.
