# ai-solutions

A public repository of technical solutions and experiments for AI.

## Repository Layout

```
ai-solutions/
├── <solution-name>/        # Stable, publishable solutions
│   └── ...
└── _experiments/           # Work-in-progress; not yet ready for wide publishing
    └── <solution-name>/
        └── ...
```

Each solution lives in its own self-contained directory. A solution moves from `_experiments/` to the top level once it is considered stable and ready for wider use.

## Conventions

- **Solution directory names** use `snake_case` (e.g., `llama_cpp_gemm_stats`).
- Keep each solution self-contained — avoid cross-solution dependencies unless they are clearly documented.
- Prefer small, focused solutions over monolithic ones.

## Standard Files per Solution

| File | Required | Purpose |
|------|----------|---------|
| `README.md` | Yes | What the solution does, how to build/run it, design notes |
| `Makefile` | Yes | Utility targets for building, running, and environment setup |
| `CLAUDE.md` | When needed | Claude Code guidance: architecture overview, key files, dev workflow, invariants — add when the solution is complex enough that these aren't obvious from reading the code |
| `.gitignore` | When needed | Exclude build artifacts, venv dirs, generated files, etc. |

### Makefile Guidelines

Use `make` as the single entry point for all common tasks. Typical targets:

- **`make setup`** — create and populate a Python venv (`python -m venv .venv && .venv/bin/pip install -r requirements.txt`) or build a Docker image.
- **`make run`** — run the solution (activating the venv or invoking `docker run` as appropriate).
- **`make build`** — compile or package if needed.
- **`make clean`** — remove generated files, the venv, or Docker artifacts.

When both a venv and Docker are useful, provide separate targets (e.g., `make run` for local venv, `make docker-run` for the container).

### Environment Setup

- **Python venv** — use `.venv/` inside the solution directory; add `.venv/` to `.gitignore`.
- **Docker** — write a `Dockerfile` in the solution directory and manage image build/run through `make` targets.

## Adding a New Solution

1. Start under `_experiments/<solution-name>/` while the work is exploratory.
2. Add `README.md` and `Makefile` inside the directory from day one.
3. When the solution is stable and well-documented, move it to `<solution-name>/` at the repo root and update any cross-references.
