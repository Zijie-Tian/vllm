# Agent Instructions for vLLM

> These instructions apply to **all** AI-assisted contributions to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Model evaluation results when the change affects output, accuracy, or serving.
    - Clear statement that AI assistance was used.
- Before opening a PR (drafts included) or requesting re-review, run the [`pr-checklist`](.agents/skills/pr-checklist/SKILL.md) skill and address its findings.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3`, bare `pip`, `uv venv`, or `.venv`.** Every test, smoke check, and native build in this checkout uses the Conda environment `mxmoe`.
- Follow the parent [unified environment guide](../../doc/environment.md) for both accuracy and performance. Run `conda run --no-capture-output -n mxmoe python ...`, or activate `mxmoe` first; apply the parent `constraints.txt` to additional dependency installs.

### Environment setup

```bash
conda run --no-capture-output -n mxmoe python -m pip install -c ../../constraints.txt -r requirements/lint.txt
conda run --no-capture-output -n mxmoe pre-commit install
```

### Installing dependencies

Install the shared dependencies using the parent guide first. Keep vLLM as a source-editable install in `mxmoe`, not a precompiled package or a second environment. With the guide's toolchain and source-cache variables set, rebuild with:

```bash
conda run --no-capture-output -n mxmoe python -m pip install --no-build-isolation --no-deps --editable .
```

For C/C++ or CUDA changes, follow the
[incremental compilation workflow](docs/contributing/incremental_build.md).

### Tests

```bash
conda run --no-capture-output -n mxmoe python -m pip install -c ../../constraints.txt -r requirements/test/cuda.in
conda run --no-capture-output -n mxmoe python -m pytest tests/path/to/test_file.py -v
```

When adding tests:

- **Design before you write.** Answer four questions first: what is the module
  for, what is its I/O contract, what failure am I guarding against, and what is
  the cheapest level that catches it (unit over integration over e2e)?
- **Reuse before create.** Extend existing test files, `conftest.py` fixtures, and
  helpers; add a new file only when no nearby suite fits.
- **Test behavior with intent.** Assert observable outcomes through public APIs;
  state why in the name or docstring. Skip trivial wiring; flaky tests are worse
  than no tests.
- **Keep it minimal.** One behavior per test and the smallest setup that
  triggers it; if the test diff dwarfs the code change, cut scope.
- **No one-off kernel benchmarks in `tests/`.** Put kernel perf work in
  `benchmarks/kernels/`; prove correctness in existing pytest suites.
- **Run model evals for model-affecting changes.** Search `tests/evals/` or use
  `vllm bench` and include results in the PR — do not wait for reviewers to ask.

For model-specific requirements, see
[`docs/contributing/model/tests.md`](docs/contributing/model/tests.md).

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
conda run --no-capture-output -n mxmoe pre-commit run

# Run on all files:
conda run --no-capture-output -n mxmoe pre-commit run --all-files

# Run a specific hook:
conda run --no-capture-output -n mxmoe pre-commit run ruff-check --all-files
```

The line length limit for Python code is 88 characters. If you are not sure, use pre-commit to check.

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`/`Returns:`/`Raises:` sections), not reStructuredText/Sphinx fields (`:param:`, `:return:`, `:rtype:`).

### Coding style guidelines

- Match existing code style
- Minimize use of comments. Eliminate comments which are redundant, preferring legible and self-documenting code. When used, keep docstrings and comments brief and direct.
- Assume the reader is familiar with vLLM.

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`):

```text
Your commit message here

Co-authored-by: Agent Name Here
Signed-off-by: Your Name <your.email@example.com>
```

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

Security reviewers should start with [`SECURITY.md`](SECURITY.md),
[`docs/usage/security.md`](docs/usage/security.md), and
[`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)
for the project security policy, threat model, deployment assumptions, and
vulnerability process.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.
