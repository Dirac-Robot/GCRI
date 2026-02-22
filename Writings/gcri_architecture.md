# GCRI: Architecture & Design Philosophy

> Generalized Cognitive Refinement Iteration — a multi-branch adversarial reasoning framework for LLM agents.

---

## 1. Core Philosophy: Falsification over Self-Reflection

GCRI rejects the dominant self-reflection paradigm (Reflexion, Self-Refine) in favor of **Popperian falsification**. The fundamental insight:

- **Self-reflection asks:** *"Is this correct? What can be improved?"* — an open-ended judgment task that LLMs perform poorly (Huang et al., ICLR 2024: "LLMs Cannot Self-Correct Reasoning Yet").
- **Falsification asks:** *"Find a specific input where this fails."* — a concrete search task that LLMs perform well.

**Three structural advantages over self-reflection:**

| Dimension | Self-Reflection | GCRI Falsification |
|---|---|---|
| **Cognitive task** | Open-ended quality judgment | Targeted counter-example search |
| **Feedback type** | Natural-language critique (ambiguous) | Concrete counter-examples (actionable) |
| **Evaluator stance** | Cooperative (wants success) → sycophancy | Adversarial (wants to break) → robustness |

The system treats hypotheses as scientific claims: they are **not confirmed by passing tests**, but **provisionally accepted by surviving adversarial attack**.

---

## 2. Execution Model: Competitive Survival

GCRI is **not** collaborative refinement. It is **competitive survival**. Hypotheses do not receive gentle feedback — they are stress-tested. If they break, they die. If they survive, they face judgment.

### 2.1 Single-Iteration Pipeline

```
┌─────────────────┐
│ Strategy        │  Analyzes task → generates N diverse approaches.
│ Generator       │  Respects Strategy Graveyard (never repeats failed approaches).
└───────┬─────────┘
        │ N strategies
        ▼
┌─────────────────┐
│ Hypothesis      │  Each strategy → isolated Docker sandbox.
│ Generator       │  Parallel execution. Code tools available.
│ (per branch)    │  Optional Reasoning Agent refines before verification.
└───────┬─────────┘
        │ N raw hypotheses
        ▼
┌─────────────────┐
│ Hypothesis      │  Merges/filters branches. Intelligent code-level merging.
│ Aggregator      │  Output: M ≤ N aggregated branches for verification.
└───────┬─────────┘
        │ M aggregated branches
        ▼
┌─────────────────┐
│ Verification    │  RED TEAM. Tries to BREAK each hypothesis.
│ Agent           │  Finds concrete counter-examples. Fixes if possible.
│ (per branch)    │  Output: counter_example + counter_strength (strong/weak/none)
└───────┬─────────┘
        │ M verified results
        ▼
┌─────────────────┐
│ Decision        │  Evaluates all branches. Accepts or rejects ALL.
│ Maker           │  If accepted: selects best branch, extracts final output.
│                 │  If rejected: synthesizes global feedback for next iteration.
└───────┬─────────┘
        │ (if rejected)
        ▼
┌─────────────────┐
│ Memory          │  Converts failures → Active Constraints (permanent rules).
│ Manager         │  Failed strategies → Strategy Graveyard (never repeated).
│                 │  Curates sandbox for cross-iteration artifact preservation.
└─────────────────┘
```

### 2.2 Multi-Iteration Loop

The outer `__call__` method runs the above pipeline up to `max_iterations` times:

```
for iteration in range(max_iterations):
    result = workflow.invoke(state)    # single-iteration pipeline
    if result.decision == True:
        commit winning branch → done
    else:
        inject memory → next iteration
```

Each iteration carries forward:
- **Active Constraints**: hard rules extracted from failed counter-examples
- **Strategy Graveyard**: approaches that were tried and failed
- **Iteration History**: compressed logs of past attempts
- **Base Sandbox**: merged artifacts from promising branches

---

## 3. Agent Roles & Prompt Design

Each agent role has a dedicated prompt template under `gcri/templates/v0.1.1/`. All templates are prepended with `global_rules.txt` at runtime.

### 3.1 Strategy Generator (`strategy_generator.txt`)

**Input:** task, feedback (from previous iteration), locked_intent, num_strategies  
**Output:** `Strategies` schema — list of `Strategy` objects + intent_analysis + strictness level

Key design decisions:
- **Strictness classification**: strict (math/code), moderate (design), creative (writing)
- **Feedback handling**: absolute constraints → hints, strategy graveyard → exclusion, last errors → explicit address
- **Diversity enforcement**: varies reasoning style (constructive/adversarial, top-down/bottom-up)
- **Contrarian rule**: if graveyard shows a pattern, challenge the core assumption

### 3.2 Hypothesis Generator (`hypothesis.txt`)

**Input:** task, strategy, intent_analysis, memory, feedback  
**Output:** `Hypothesis` schema — the candidate solution

Executes inside an isolated Docker container with full code tools. The hypothesis is the actual deliverable (code files, proofs, etc.), not a description of one.

### 3.3 Reasoning Agent (`reasoning.txt`)

**Input:** task, strategy, hypothesis  
**Output:** `Reasoning` schema — refined_hypothesis + reasoning

Lightweight pre-verification refinement. Identifies logical flaws and improves clarity without changing the solution family. Optional in `LowThinkGenerator` mode.

### 3.4 Verification Agent (`verification.txt`) — THE CORE

**Input:** task, strategy, hypothesis, reasoning, intent_analysis  
**Output:** `Verification` schema:
- `counter_example`: specific failure scenario
- `counter_strength`: strong | moderate | weak | none
- `reasoning`: why the counter-example is valid
- `adjustment`: minimal fix log

**Critical prompt rules:**
1. **Logical analysis BEFORE code execution** — execution success ≠ correctness
2. **Generator's tests are NOT proof** — must independently verify against task intent
3. **On failure: FIX, don't just report** — never defer correction
4. **Never redesign from scratch** — minimal edit, preserve strategy

**Counter-strength levels:**
- `strong`: fatal flaw, execution failure, scope mismatch → branch rejected
- `weak`: minor issue, edge case → branch may survive
- `none`: no issues found → branch passes

### 3.5 Decision Maker (`decision.txt`)

**Input:** task, intent_analysis, aggregated_result, file_contexts, failure_categories  
**Output:** `DecisionProtoType` schema:
- `decision`: bool (True = at least one valid branch)
- `best_branch_index`: 0-based index of winning branch
- `branch_evaluations`: per-branch `BranchAnalysis` (status + failure_category + reasoning)
- `global_feedback`: strategic direction for next iteration (if rejected)

**Key constraints:**
- Never rewrite or merge hypotheses — judge AS-IS
- Strong counter-example unfixed → automatic rejection
- Code outputs referenced via paths only, never inlined

### 3.6 Memory Manager (`memory.txt` + `active_memory.txt`)

Structures accumulated knowledge as:
- **Active Constraints**: `new_active_constraints` — rules extracted from feedback, never violated
- **Strategy Graveyard**: failed approaches never repeated
- **Iteration History**: compressed `IterationLog` entries
- **Base Sandbox**: merged artifacts from selected branches for next iteration

---

## 4. Data Flow: Schemas & States

### 4.1 Core Schemas (`gcri/graphs/schemas.py`)

| Schema | Purpose |
|---|---|
| `Strategy` | name, description, feedback_reflection, hints |
| `Strategies` | list of Strategy + intent_analysis + strictness |
| `Hypothesis` | candidate solution text |
| `Reasoning` | refined_hypothesis + evaluation reasoning |
| `Verification` | counter_example, counter_strength, adjustment, reasoning |
| `RawHypothesis` | pre-aggregation output (index, strategy, hypothesis, reasoning, container_id) |
| `AggregatedBranch` | post-aggregation (index, source_indices, combined_hypothesis, merge_reasoning, container_id) |
| `AggregationResult` | list of AggregatedBranch + discarded_indices + summary |
| `ActiveConstraints` | list of hard rules extracted from failures |
| `DecisionProtoType` | decision bool, best_branch_index, evaluations, global_feedback |
| `FailureCategory` | enum: none, logic_error, req_missing, hallucination, practicality, other |

### 4.2 State Objects (`gcri/graphs/states.py`)

| State | Scope | Key Fields |
|---|---|---|
| `TaskState` | Global per task | task, count, strategies, raw_hypotheses, aggregated_result, decision, feedback, memory |
| `BranchState` | Per hypothesis branch | index, strategy, hypothesis, reasoning, container_id |
| `VerificationBranchState` | Per verification branch | index, aggregated_branch, container_id |
| `StructuredMemory` | Persistent across iterations | active_constraints, graveyard, iteration_history, base_sandbox_container |
| `HypothesisResult` | Post-verification | strategy, hypothesis, counter_example, counter_strength, adjustment |

---

## 5. Code Architecture

### 5.1 Module Map

```
gcri/
├── graphs/
│   ├── gcri_unit.py      # GCRI class — LangGraph workflow orchestrator
│   ├── generators.py     # BranchesGenerator variants (Default/DeepThink/LowThink)
│   ├── aggregator.py     # HypothesisAggregator — merges/filters branches
│   ├── schemas.py        # Pydantic schemas for all structured outputs
│   ├── states.py         # LangGraph state definitions
│   └── callbacks.py      # GCRICallbacks interface (CLI/Web/Auto/NoCommit)
├── tools/
│   ├── cli.py            # Tool definitions, CodeAgentBuilder, InteractiveToolGuard
│   ├── utils.py          # SandboxManager — Docker container lifecycle
│   └── docker_sandbox.py # Low-level Docker operations
├── config.py             # ato scope configuration (default/no_reasoning views)
└── templates/v0.1.1/     # Agent prompt templates
```

### 5.2 LangGraph Workflow (`gcri_unit.py`)

The main graph is a `StateGraph(TaskState)` with nodes:

```
START → generate_branches → aggregate_hypotheses
      → (map) verification_executor → collect_verification
      → decision → (conditional) update_memory → END
                                   └──────────→ END (if accepted)
```

**Critical design pattern:** Verification branches use `Send()` for parallel execution via LangGraph's map-reduce. Each branch gets its own `VerificationBranchState` with an isolated Docker container.

### 5.3 BranchesGenerator Variants (`generators.py`)

| Generator | Pipeline | Use Case |
|---|---|---|
| `DefaultBranchesGenerator` | strategy → hypothesis → reasoning | Standard balanced mode |
| `DeepThinkGenerator` | strategy → hypothesis → reasoning (extended) | Maximum quality |
| `LowThinkGenerator` | strategy → hypothesis (skip reasoning) | Speed-optimized |

All generators share `BaseBranchesGenerator` with common helpers: `_check_abort`, `_load_template_with_rules`, `_invoke_with_retry`, `_sample_strategies`.

### 5.4 Callbacks (`callbacks.py`)

Environment-agnostic event interface:

| Callback | Variants |
|---|---|
| `GCRICallbacks` (base) | No-op defaults |
| `CLICallbacks` | loguru logging + interactive commit prompt |
| `AutoCallbacks` | Auto-approve everything (benchmarks) |
| `NoCommitCallbacks` | Reject all commits (dry-run) |
| `CoBrAGCRICallbacks` | SSE events via queue (Web UI) |

### 5.5 Configuration (`config.py`)

Uses `ato.scope` for layered configuration:

- **Default view**: reasoning models (gpt-5.2 for planner/decision, gpt-5-mini for branches)
- **`no_reasoning` view**: non-reasoning models (gpt-4.1 / gpt-4.1-mini, max_tokens instead of max_completion_tokens)

Each agent is configured independently with `model_id`, `parameters`, and `gcri_options` (use_code_tools, use_web_search, max_recursion_depth).

---

## 6. Sandbox Isolation

Every hypothesis branch runs in its own Docker container (`SandboxManager` in `utils.py`):

- **Branch setup**: `setup_branch(iteration, index)` → fresh container with project files
- **Container lifecycle**: create → execute → verify → merge or discard
- **Cross-iteration persistence**: winning branch's container can be committed to host or cloned as base sandbox
- **Aggregation merging**: `HypothesisAggregator` can merge files from multiple containers via LLM-guided intelligent merge

---

## 7. CoBrA Integration

`GCRIWithCoMeT` (in CoBrA's `gcri_comet.py`) extends GCRI:

- **CoMeT memory tools** injected into all branch agents (retrieve_knowledge, read_knowledge)
- **Web search** results auto-saved to CoMeT
- **Verified results** ingested into CoMeT as trusted knowledge
- **Token tracking** via `TokenTracker` callback
- **Provider resolution**: auto-detects model provider (openai/anthropic/google_genai) from model_id prefix
- **Build model patching**: monkey-patches `build_model` and `build_decision_model` to inject CoMeT tools + tracker

---

## 8. Key Design Principles

1. **Asymmetric evaluation**: Generator optimizes for correctness; Verifier optimizes for destruction. Different objectives break echo chambers.
2. **Structural feedback**: Counter-examples are concrete and actionable, unlike natural-language critique.
3. **Failure as knowledge**: Failed counter-examples become Active Constraints; failed strategies enter the Graveyard. Nothing is wasted.
4. **Isolation guarantees**: Docker containers ensure branches cannot interfere. File-level merging happens only through explicit aggregation.
5. **Environmental agnosticism**: Callbacks decouple GCRI from any specific runtime (CLI, Web, API, benchmark harness).
6. **Configuration composability**: `ato.scope` views allow switching between reasoning/non-reasoning models without code changes.
