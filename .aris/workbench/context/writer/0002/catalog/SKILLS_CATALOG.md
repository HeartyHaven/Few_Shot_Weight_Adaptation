> Navigation snapshot from `/home/lang_gao/.cache/superaris/aris/8f493923f28c3f991dcca1287db97aa780d0969b/docs/SKILLS_CATALOG.md`. Selected skill links resolve to this role's copies; other skill links lead to on-demand preparation. Descriptions of external tools do not establish their availability.

# ARIS Skills Catalog

Every skill that ships with ARIS, grouped by role. **87 skills** as of the
latest update; new skills land via PR and get added to the table below.

- Each `Skill` link goes to the canonical `SKILL.md` (the LLM-readable spec).
- `Role` is a one-line summary — see the `SKILL.md` for the full contract,
  phases, and triggers.
- `Requires` lists external dependencies beyond ARIS core (Codex MCP, Gemini
  API, Modal account, LaTeX toolchain, etc.). `None` means it works out of
  the box on a standard install.

> **Codex CLI mirror:** every skill below has a parallel implementation
> under [`skills/skills-codex/`](../navigation.md#capabilities-and-tools) for Codex CLI users.
> The mirror swaps the Codex-MCP reviewer path for Codex-native
> `spawn_agent` + `send_input`. SKILL semantics are otherwise identical;
> the table below tracks the main-tree canonical files.

---

## 🏗️ Workflow Orchestrators

End-to-end pipelines that chain many sub-skills. Most users start here.

| Skill | Role | Requires |
|---|---|---|
| [`/research-pipeline`](../navigation.md#on-demand-skills) | **Full chain** — Workflow 1 → 1.5 → 2 → 3, from research direction to submission-ready paper | Codex MCP, LaTeX, GPU |
| [`/idea-discovery`](../navigation.md#on-demand-skills) | **Workflow 1** — research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline | Codex MCP |
| [`/idea-discovery-robot`](../navigation.md#on-demand-skills) | Workflow 1 adapter for robotics / embodied AI — robotics-aware literature survey + benchmark-anchored ideation | Codex MCP |
| [`/experiment-bridge`](../navigation.md#on-demand-skills) | **Workflow 1.5** — read experiment plan → implement code → sanity check → deploy to GPU → collect initial results | GPU (local / remote / Vast / Modal) |
| [`/auto-review-loop`](../navigation.md#on-demand-skills) | **Workflow 2** — autonomous review → fix → re-review until positive or max rounds; uses Codex MCP reviewer | Codex MCP |
| [`/auto-review-loop-llm`](../navigation.md#on-demand-skills) | Same as Workflow 2 but uses any OpenAI-compatible LLM via [`llm-chat`](../navigation.md#capabilities-and-tools) MCP server | llm-chat MCP |
| [`/auto-review-loop-minimax`](../navigation.md#on-demand-skills) | Workflow 2 variant pinned to MiniMax API | MiniMax API key |
| [`/paper-writing`](../navigation.md#on-demand-skills) | **Workflow 3** — paper-plan → paper-figure → illustration → paper-write → paper-compile → auto-paper-improvement-loop | Codex MCP, LaTeX |
| [`/rebuttal`](../navigation.md#on-demand-skills) | **Workflow 4** — parse reviews → atomize → strategy → draft → safety check → stress test → 2-version output → follow-ups | Codex MCP |
| [`/resubmit-pipeline`](../navigation.md#on-demand-skills) | **Workflow 5** — text-only port across venues (no new experiments, no bib edits) — isolation → anonymity → audits `--soft-only` → microedit → kill-argument gate → compile + push | Codex MCP, LaTeX |
| [`/paper-talk`](../navigation.md#on-demand-skills) | **Workflow 6** — paper → slide outline → Beamer + PPTX → per-page polish → assurance audits → final report | Codex MCP, LaTeX, python-pptx |
| [`/research-refine-pipeline`](../navigation.md#on-demand-skills) | Sub-pipeline used by `/idea-discovery` — refine method + plan experiments in one chain | Codex MCP |
| [`/patent-pipeline`](../navigation.md#on-demand-skills) | Full patent drafting — invention → claims → spec → jurisdiction format (CN / US / EP) | Codex MCP |
| [`/dse-loop`](../navigation.md#on-demand-skills) | Autonomous design-space exploration loop for computer architecture / EDA — run → analyze → tune → iterate until objective met | Domain-specific tools |
| [`/meta-optimize`](../navigation.md#on-demand-skills) | **Workflow M** — analyze ARIS usage logs and propose SKILL.md / prompt / default-parameter improvements (outer-loop self-evolution) | Codex MCP, hook logging |
| [`/meta-apply`](../navigation.md#on-demand-skills) | **Privileged landing gate** — the only skill allowed to mutate the skill corpus; lands `/meta-optimize` patches the human approved, after a fresh cross-model jury PASS on the staged diff (read-only producer ≠ privileged applier) | Codex MCP, human-in-loop |

## 📚 Literature & Search

Paper retrieval, summarization, novelty verification.

| Skill | Role | Requires |
|---|---|---|
| [`/research-lit`](../navigation.md#on-demand-skills) | Multi-source literature search — Zotero / Obsidian / local PDFs / web / arXiv / S2 / DeepXiv / Exa / Gemini / OpenAlex with cross-source dedup | None (sources gated by MCP / SDK availability) |
| [`/arxiv`](../navigation.md#on-demand-skills) | Search, download, summarize arXiv papers; multi-result table + per-paper detail | None |
| [`/semantic-scholar`](../navigation.md#on-demand-skills) | Published-venue paper search (IEEE / ACM / Springer) — citation counts, venue metadata, TLDR | None (rate-limited without S2 API key) |
| [`/deepxiv`](../navigation.md#on-demand-skills) | Progressive paper reading — search → brief → head → section → trending → web search | `pip install deepxiv-sdk` |
| [`/exa-search`](../navigation.md#on-demand-skills) | AI-powered broad web search with content extraction — blogs, docs, news, papers | `pip install exa-py` + `EXA_API_KEY` |
| [`/keenable-search`](../navigation.md#on-demand-skills) | Keyless broad web search with page-text snippets, plus fetching a known URL as markdown (blogs, docs, project pages, news) | None (optional `KEENABLE_API_KEY` raises rate limits) |
| [`/firecrawl-search`](../navigation.md#on-demand-skills) | Broad web search with full-page markdown, plus scraping a known URL (HTML or PDF) to markdown — blogs, docs, project pages, technical reports | `FIRECRAWL_API_KEY` |
| [`/web-debug-search`](../navigation.md#on-demand-skills) | Multi-source debugging search across GitHub, Stack Exchange, Chinese technical communities, and general web — routing, compatibility, and discovery-only results | None |
| [`/openalex`](../navigation.md#on-demand-skills) | OpenAlex API search — 250M+ open citation graph, institutional affiliations, funding data | `pip install requests` |
| [`/gemini-search`](../navigation.md#on-demand-skills) | Gemini-driven literature discovery — decomposes topics into sub-problems, aliases, variants | `gemini-cli` v0.40+ |
| [`/alphaxiv`](../navigation.md#on-demand-skills) | Quick single-paper lookup via [AlphaXiv](https://www.alphaxiv.org) — three-tier fallback (overview → markdown → LaTeX source) | None |
| [`/comm-lit-review`](../navigation.md#on-demand-skills) | Communications-domain literature review with Claude-style knowledge-base-first retrieval — wireless / networking / satellite / Wi-Fi / cellular | None |
| [`/novelty-check`](../navigation.md#on-demand-skills) | Verify a research idea is novel against recent literature — multi-source search + cross-model verification + closest-prior-work table | Codex MCP |

## 💡 Ideation & Method Design

Generating, refining, planning research ideas before implementation.

| Skill | Role | Requires |
|---|---|---|
| [`/idea-creator`](../navigation.md#on-demand-skills) | Brainstorm 8-12 ideas, filter by feasibility, pilot on GPU, rank by signal | Codex MCP, GPU for pilots |
| [`/research-refine`](../navigation.md#on-demand-skills) | Iterative method refinement — problem anchor → up to 5 review rounds → score ≥ 9 | Codex MCP |
| [`/experiment-plan`](../navigation.md#on-demand-skills) | Turn a refined proposal into a claim-driven experiment roadmap — ablations, budgets, run order | None |
| [`/ablation-planner`](../navigation.md#on-demand-skills) | Design ablation studies from a reviewer's perspective (after main results pass `/result-to-claim`) | Codex MCP |
| [`/formula-derivation`](../navigation.md#on-demand-skills) | Structure theory derivations — organize assumptions, build derivation chains, turn scattered equations into coherent narrative | None |

## 🧠 Proof Engineering

Orchestrating and independently auditing mathematical proofs.

| Skill | Role | Requires |
|---|---|---|
| [`/proof-orchestrator`](../navigation.md#on-demand-skills) | Stateful proof-run orchestration: run directories, cross-run continuation, manual GPT Pro handoff packages, optional DeepSeek second opinion (additional evidence only — `/proof-writer` drafts proofs, `/proof-checker` owns verification and submission acceptance) | None by default; llm-chat MCP only for optional DeepSeek review |
| [`/lean-formalize`](../navigation.md#on-demand-skills) | Formalize, continue, or audit a Lean proof when requested or useful for a substantial mathematical obligation; preserve the original statement, connect actual inputs, and verify the final declaration and its axioms | Lean project/toolchain; authorized cross-family reviewer for substantial proof checkpoints |

## 🧪 Experiments & Infrastructure

GPU job submission, scheduling, monitoring, profiling.

| Skill | Role | Requires |
|---|---|---|
| [`/research-implement-feature`](../navigation.md#on-demand-skills) | Build from a plain "implement X" request — runnable spine first, then one feature per rung; every under-determined decision declared in an assumption ledger before the code depends on it, plus a cross-model sweep for the ones that slipped through undeclared | Codex MCP |
| [`/run-experiment`](../navigation.md#on-demand-skills) | Deploy experiments to local / remote / Vast.ai / Modal GPU | GPU (configurable) |
| [`/monitor-experiment`](../navigation.md#on-demand-skills) | Monitor running experiments, check progress, collect results | None |
| [`/analyze-results`](../navigation.md#on-demand-skills) | Compute statistics, generate comparison tables, surface insights from experiment results | None |
| [`/experiment-queue`](../navigation.md#on-demand-skills) | SSH job queue for multi-seed / multi-config sweeps — OOM retry, stale-screen cleanup, wave gating, crash-safe state | SSH access |
| [`/vast-gpu`](../navigation.md#on-demand-skills) | Rent, manage, destroy on-demand GPU on [Vast.ai](https://vast.ai) | Vast.ai account + `vast-cli` |
| [`/serverless-modal`](../navigation.md#on-demand-skills) | Run GPU workloads on [Modal](https://modal.com) — zero-config serverless, auto scale-to-zero | `pip install modal` + Modal account |
| [`/qzcli`](../navigation.md#on-demand-skills) | Manage GPU compute jobs on the Qizhi (启智) platform via `qzcli` (kubectl-style CLI) | `qzcli` installed |
| [`/training-check`](../navigation.md#on-demand-skills) | Periodically poll W&B metrics during training — catch NaN, loss divergence, idle GPUs early | W&B account |
| [`/system-profile`](../navigation.md#on-demand-skills) | Profile a target (script / process / GPU / memory / interconnect) with external tools + code instrumentation; produce actionable report | Profiling tools |

## 🛡️ Review, Audit & Assurance

Cross-model critique, integrity checking, evidence verification.

| Skill | Role | Requires |
|---|---|---|
| [`/research-review`](../navigation.md#on-demand-skills) | Single-round deep critical review from external LLM (Codex GPT xhigh by default; `oracle-pro` route for Pro tier) | Codex MCP (or Oracle MCP) |
| [`/experiment-audit`](../navigation.md#on-demand-skills) | Cross-model integrity check of experiment code + results — catches fake ground truth, score-normalization fraud, phantom results, scope inflation | Codex MCP |
| [`/result-to-claim`](../navigation.md#on-demand-skills) | Map experimental results to intended claims — judges what's supported, what's not, what's missing; routes to next action | Codex MCP |
| [`/paper-claim-audit`](../navigation.md#on-demand-skills) | Zero-context numeric verification — every number / comparison / scope claim in the paper checked against raw result files by a fresh reviewer (no confirmation bias) | Codex MCP |
| [`/citation-audit`](../navigation.md#on-demand-skills) | Bibliography audit — existence + metadata correctness + context appropriateness for every `\cite{}`; `--soft-only` mode for frozen-bib resubmits | Codex MCP, web access |
| [`/citation-crosscheck`](../navigation.md#on-demand-skills) | Double-blind bib-vs-web field audit — one agent extracts each cited entry verbatim from the local `.bib`, a second re-fetches it from the web blind to the `.bib`, a third emits a mechanical field-by-field delta (author set+order, year, venue, arXiv-id/DOI); a deterministic `cx_verify.py` gate assigns MATCH/MINOR and fails closed to the main agent's primary-source fetch; side-by-side `.xlsx` per entry | web access; openpyxl (for the `.xlsx`) |
| [`/proof-checker`](../navigation.md#on-demand-skills) | Rigorous mathematical proof verification — 20-category issue taxonomy, two-axis severity, side-condition checklists, counterexample red team, proof-obligation ledger | Codex MCP |
| [`/kill-argument`](../navigation.md#on-demand-skills) | Two-thread adversarial review — Thread 1 writes the strongest 200-word rejection memo; Thread 2 (independent) defends point-by-point and surfaces still-unresolved issues | Codex MCP |
| [`/integrity-forensics`](../navigation.md#on-demand-skills) | SHA-pinned thin launcher for [Anti-Autoresearch](https://github.com/wanshuiyin/Anti-Autoresearch) — evidence-ledger forensic sweep (46 patterns, deterministic adjudicator) → typed BLOCK/WARN/NO_NEW_BLOCKER gate + append-only obligations ledger; default pre-submission self-audit in `/paper-writing` | git, Codex MCP (via upstream) |

## 📝 Paper Writing & Figures

LaTeX generation, figure / diagram production, prose polishing.

| Skill | Role | Requires |
|---|---|---|
| [`/paper-plan`](../navigation.md#on-demand-skills) | Generate a structured paper outline from review conclusions + experiment results — claims-evidence matrix, section structure, figure plan, citation scaffolding | None |
| [`/paper-write`](../navigation.md#on-demand-skills) | Section-by-section LaTeX generation (ICLR / NeurIPS / ICML / IEEE / ACL / AAAI / CVPR / ACM MM). Anti-hallucination BibTeX via DBLP / CrossRef | None |
| [`/paper-figure`](../navigation.md#on-demand-skills) | Publication-quality matplotlib / seaborn plots + LaTeX comparison tables from experiment results | matplotlib / seaborn |
| [`/figure-spec`](../navigation.md#on-demand-skills) | Deterministic JSON → SVG renderer for architecture / workflow / pipeline / audit-cascade diagrams. Shape-aware edge clipping, self-loops, CJK width estimation | None |
| [`/paper-illustration`](../navigation.md#on-demand-skills) | AI architecture + method illustrations via Gemini image generation, with Claude-supervised iterative refinement | `GEMINI_API_KEY` |
| [`/paper-illustration-image2`](../navigation.md#on-demand-skills) | Codex-native image generation alternative — uses ChatGPT Plus / Pro quota via local Codex app-server bridge (no Gemini key) | Codex app-server + `codex-image2` MCP bridge |
| [`/mermaid-diagram`](../navigation.md#on-demand-skills) | Generate Mermaid diagrams from requirements — flowcharts, sequence, class, ER, Gantt, with syntax verification | None |
| [`/pixel-art`](../navigation.md#on-demand-skills) | Generate pixel-art SVG illustrations for READMEs, docs, slides | None |
| [`/paper-compile`](../navigation.md#on-demand-skills) | Compile LaTeX paper to PDF — auto-fix errors, submission readiness checks | LaTeX (`latexmk`, `pdfinfo`) |
| [`/auto-paper-improvement-loop`](../navigation.md#on-demand-skills) | 2-round content review + format check — typical 4 / 10 → 8.5 / 10 score lift. `--edit-whitelist` mode for resubmits | Codex MCP |
| [`/proof-writer`](../navigation.md#on-demand-skills) | Draft rigorous mathematical proofs for ML / AI theory — theorems, lemmas, propositions, corollaries; fill in missing steps; formalize sketches | None |
| [`/writing-systems-papers`](../navigation.md#on-demand-skills) | Paragraph-level structural blueprint for 10-12 page systems papers — page allocation, paragraph templates, writing patterns for OSDI / SOSP / ASPLOS / NSDI / EuroSys | None |
| [`/grant-proposal`](../navigation.md#on-demand-skills) | Structured grant proposal drafting — KAKENHI (JP), NSF (US), NSFC (CN including 面上 / 青年 / 优青 / 杰青 / 海优 / 重点), ERC (EU), DFG (DE), more | None |

## 🎤 Talks, Posters & Resubmission

After-paper outputs and venue porting.

| Skill | Role | Requires |
|---|---|---|
| [`/paper-slides`](../navigation.md#on-demand-skills) | Conference presentation — Beamer LaTeX → PDF + editable PPTX + speaker notes + full talk script | LaTeX, python-pptx |
| [`/slides-polish`](../navigation.md#on-demand-skills) | Per-page Codex review + targeted python-pptx / Beamer fixes (font scaling, frame resize, banner-as-tcolorbox, italic leak guard, em-dash spacing, CJK font hint, anonymity placeholder discipline) | Codex MCP, python-pptx |
| [`/paper-poster-html`](../navigation.md#on-demand-skills) | **Default** conference poster — single HTML/CSS file with measurement-driven hard gates (two-hue tokens, real paper figures with provenance, anti-patch-loop fix vocabulary) → print-ready PDF via headless Chromium | Playwright (Chromium), PyMuPDF |
| [`/paper-poster`](../navigation.md#on-demand-skills) | DEPRECATED — redirect stub to `/paper-poster-html` (legacy LaTeX pipeline retired; in git history) | — |

(Orchestrators `/paper-talk` for the talk pipeline and `/resubmit-pipeline`
for venue porting live under [Workflow Orchestrators](#%EF%B8%8F-workflow-orchestrators).)

## 📜 Patents

End-to-end patent drafting and prior-art workflow.

| Skill | Role | Requires |
|---|---|---|
| [`/invention-structuring`](../navigation.md#on-demand-skills) | Structure a raw invention idea into a formal invention disclosure | None |
| [`/claims-drafting`](../navigation.md#on-demand-skills) | Draft patent claims — independent + dependent, with anti-pattern checks | None |
| [`/embodiment-description`](../navigation.md#on-demand-skills) | Write detailed embodiment descriptions for the patent specification | None |
| [`/specification-writing`](../navigation.md#on-demand-skills) | Full patent specification from claims + invention disclosure | None |
| [`/figure-description`](../navigation.md#on-demand-skills) | Generate formal drawing descriptions for patent figures | None |
| [`/prior-art-search`](../navigation.md#on-demand-skills) | Search patent databases + academic literature for prior art relevant to an invention | None (web access) |
| [`/patent-novelty-check`](../navigation.md#on-demand-skills) | Assess patent novelty and non-obviousness against prior art (patentability evaluation) | Codex MCP |
| [`/patent-review`](../navigation.md#on-demand-skills) | External patent-examiner-style critical review of a patent application | Codex MCP |
| [`/jurisdiction-format`](../navigation.md#on-demand-skills) | Compile patent application into jurisdiction-specific filing format (CN / US / EP) | None |

(Orchestrator `/patent-pipeline` chaining all of the above lives under
[Workflow Orchestrators](#%EF%B8%8F-workflow-orchestrators).)

## 🧰 Meta, Utilities & Integrations

Cross-cutting infrastructure used by other skills or run on demand.

| Skill | Role | Requires |
|---|---|---|
| [`/research-wiki`](../navigation.md#on-demand-skills) | Persistent research knowledge base — papers / ideas / experiments / claims with typed relationships. Workflow hooks auto-ingest across the research lifecycle | None (pure Python stdlib) |
| [`/wiki-enrich`](../navigation.md#on-demand-skills) | Fill the per-paper TODO sections that `ingest_paper` leaves as scaffolds (Karpathy LLM-wiki principle). Fetch chain alphaxiv → deepxiv → arXiv → page abstract; idempotent by default, `--force` to rewrite | Python stdlib + WebFetch |
| [`/render-html`](../navigation.md#on-demand-skills) | Render ARIS Markdown / JSON artifacts into reviewed single-file HTML views for human reading | Python stdlib; Codex MCP for review gate |
| [`/overleaf-sync`](../navigation.md#on-demand-skills) | Two-way sync between local paper directory and Overleaf project via Overleaf Git bridge (Premium) — `setup` / `pull` (diff protocol) / `push` (confirmation gate) / `status` | Overleaf Premium + macOS Keychain |
| [`/feishu-notify`](../navigation.md#on-demand-skills) | Send notifications to Feishu / Lark — push-only (webhook) or interactive (bidirectional) modes. Off by default | Feishu webhook URL |
| [`/interview-cheatsheet`](../navigation.md#on-demand-skills) | Generate long-form Chinese ML / LLM interview-prep cheat sheets with formulas, code, Q&A, review, and HTML output | Codex MCP, Python |

---

## How to use this catalog

- **Looking for a workflow entry point?** Start with [Workflow Orchestrators](#%EF%B8%8F-workflow-orchestrators).
- **Want to add a skill to an existing workflow?** Read the orchestrator's
  `SKILL.md` to see which sub-skills it composes.
- **Building your own pipeline?** Pick the skills from each category and
  chain them via prompt — no framework lock-in, every skill is a single
  `SKILL.md` readable by any LLM agent.

## Adding a new skill

1. Create `skills/<name>/SKILL.md` with `name:` + `description:` frontmatter
   (the description shows up in the LLM's slash-command autocomplete).
2. Per the
   [`integration-contract.md`](../skills/shared-references/integration-contract.md)
   §2 contract, if your skill invokes any helper script under `tools/`, use
   the canonical resolver chain — do NOT hardcode `python3 tools/foo.py`.
3. Add a row to the appropriate category table above (or propose a new
   category in your PR if your skill doesn't fit).
4. The advisory CI lint will catch any hardcoded-path regressions on PR.

See the [main README](../navigation.md#capabilities-and-tools) for installation, setup, and end-to-end
workflow examples.
