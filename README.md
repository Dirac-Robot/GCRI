# GCRI
**Generalized Counterexample-Reinforced Intelligence**

## Overview
GCRI is a lightweight LLM-based pipeline that executes multi-branch hypothesis-verification loops (generate → refine → verify → decide). It explores single tasks through multiple strategies to derive final decisions, or uses a meta-planner to execute multiple tasks sequentially.

## Quick Start

### Prerequisites
- Python 3.11+ recommended
- Install dependencies with pip (requirements.txt included)
- Set up environment variables in .env file (e.g., OPENAI_API_KEY)

### Installation

1) Create virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.\.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2) Basic execution (interactive)

**GCRI Unit Worker** (process one task with multiple strategies)

```bash
python -m gcri.entry
# or
python -c "from gcri.entry import cli_entry; cli_entry()"
```

Enter the task you want to analyze (or file path containing the task) at the terminal prompt.

**Meta-Planner Mode** (planning → delegate to multiple GCRI tasks)

```bash
python -m gcri.entry plan
```

3) Programmatic usage (simple example)

```python
from gcri.config import scope
from gcri.graphs.gcri_unit import GCRI

config = scope()  # returns config object
unit = GCRI(config)
result = unit('Enter your goal as a sentence')
print(result['final_output'])
```

## Core Use Case
- **Goal**: Explore complex problems (e.g., design/policy/debugging tasks) through multiple reasoning strategies in parallel to obtain reliable conclusions.
- **Execution**: `python -m gcri.entry` → input task → internally creates multiple branches (default 3) that cycle through hypothesis generation → refinement → verification → decision.

## Architecture Overview

### Main Components

#### gcri.config
- Scope-based configuration management and template/agent parameter declaration.

#### gcri.entry
- CLI entry point. Executes unit worker or meta-planner depending on 'plan' argument.

#### gcri.main / gcri.main_planner
- Handle unit GCRI execution loop and meta-planner execution loop respectively.

#### gcri.graphs.gcri_unit.GCRI
- Core workflow: strategy sampling → branch-wise hypothesis generation/refinement/verification → aggregation → decision → memory update.
- Main public method: `__call__(task, initial_memory=None)` → iteration result (decision/output/memory)

#### gcri.graphs.planner.GCRIMetaPlanner
- High-level planner: receives goal, decomposes into multiple single tasks, and delegates each task to GCRI unit.
- Public method: `__call__(goal)` → final state (or failure log)

#### gcri.tools.cli
- Local tool wrappers: `execute_shell_command`, `read_file`, `write_file`, `local_python_interpreter`
- `build_model(model_id, gcri_options=None, **parameters)`: internally returns CodeAgentBuilder to create agent (LLM) objects.

#### Templates (gcri/templates/*.txt)
- Prompt templates used at each stage (hypothesis, reasoning, verification, strategy_generator, decision, memory, etc.)

### Data/Schemas
- **gcri.graphs.schemas**: Defines agent outputs (hypothesis/verification/decision/plan/compression, etc.) using Pydantic models.
- **gcri.graphs.states**: Represents workflow states (Iteration, Memory, TaskState, etc.) with Pydantic.

## Public API Surface (Summary)

### Classes / Functions

#### gcri.graphs.gcri_unit.GCRI
- Usage: `GCRI(config)(task, initial_memory=None)`
- Returns: dict containing keys like 'decision', 'final_output', 'memory', etc.

#### gcri.graphs.planner.GCRIMetaPlanner
- Usage: `GCRIMetaPlanner(config)(goal)`
- Returns: dict containing 'final_answer' or planner logs.

#### gcri.tools.cli.build_model(model_id, gcri_options=None, **parameters)
- LLM agent builder. Allows local tool access based on `gcri_options.use_code_tools`.

#### gcri.tools.cli Tools
- `execute_shell_command` / `read_file` / `write_file` / `local_python_interpreter`
- Auxiliary execution tools in local environment (execution controlled via InteractiveToolGuard).

#### gcri.entry.cli_entry()
- CLI script entry point (recommended for use as entry).

### Templates/Configuration
- Templates are core to project operation (in prompt form). Default path: `gcri/templates/*.txt`.
- Configuration can be changed via Scope in `gcri/config.py`. Adjust model ID, parameters, number of branches, etc.

## Auto-Documentation Guidelines

### Recommended Automated Tools (Python code basis)

**Sphinx + autodoc** (recommended)
1. Install: `pip install sphinx sphinx-autodoc-typehints`
2. Initialize: `sphinx-quickstart docs -q`
3. conf.py: `extensions=['sphinx.ext.autodoc','sphinx.ext.napoleon','sphinx_autodoc_typehints']`
4. autodoc example: in docs/index.rst, `.. automodule:: gcri.graphs.gcri_unit`
5. Build: `sphinx-build -b html docs/ docs/_build/html`

**Alternative**: pdoc or mkdocstrings (Markdown-oriented) can be used.

### If Auto-Documentation is Not Possible (or for Quick Deployment)
- Keep the "Public API Surface" section in README up-to-date.
- Include usage examples and input/output examples for each major class (GCRI, GCRIMetaPlanner, build_model).
- TODO: Create `gcri/README_API.md` with function signatures and simple examples.

## Testing, Build, Contributing

### Testing
- Currently pytest-related output files (pytest_output.txt) exist in project root, but no separate test suite is visible.
- Recommended: Add pytest-based test directory `tests/` and run

```bash
pip install pytest
pytest -q
```

### Static Analysis
- Recommended: flake8 / pylint. Example:

```bash
pip install pylint
pylint gcri
```

### Build
- For pure Python package distribution, pyproject.toml is available. Use poetry or pip build for packaging/distribution.

### Contribution Guide (Simple)
- Fork → feature branch → PR
- Code style: black + isort recommended
- Commit messages: clearly state purpose of changes
- When adding new templates/model configurations, don't forget to update templates directory and config

### License
- Currently no LICENSE file in root. If you want open source release, add appropriate license such as MIT/Apache-2.0.

## Troubleshooting (FAQ)

### 1) Agent initialization failure (LLM model related)
- Check if required authentication keys exist in .env.
- Verify model ID and parameters in gcri/config.py are correct.

### 2) Template file not found
- Check config.templates path. Since relative paths are specified, path may differ depending on working directory.

### 3) Tool execution stops at terminal — InteractiveToolGuard prompt
- Local tools (execute_shell_command, etc.) require user confirmation before execution via interactive guard. If you want automatic execution, modify auto_mode logic inside InteractiveToolGuard, or disable tool usage (`gcri_options.use_code_tools=False`).

### 4) Logs/output not saved
- Default log directory is determined by `config.log_dir` in gcri/config.py. Check write permissions and path existence.

## Documentation TODO (Recommended)
- Add simple comments (input field descriptions) to each template (gcri/templates/*.txt).
- Document gcri/tools/cli InteractiveToolGuard behavior.
- Create README_API.md: Add examples and sample JSON return values for each function/class/module.

## Final Notes
This project combines LLM (conversational models) with a workflow engine (langgraph) to test and decide multiple reasoning paths in parallel and iteratively. Template (prompt) design determines quality, so template modification and config tuning are essential in actual use.

## Contact
Repository maintainers or code contributors should add contact information/issue templates through README.
