# ai-solutions

A collection of technical solutions and experiments around AI — covering topics such as inference optimization, tooling integrations, benchmarking, and other practical explorations.

## Structure

```
ai-solutions/
├── <solution-name>/        # Stable, well-documented solutions
│   ├── README.md
│   ├── Makefile
│   └── ...
└── _experiments/           # Work-in-progress; not yet ready for wider use
    └── <solution-name>/
        └── ...
```

Each solution is self-contained in its own directory with a `README.md` explaining what it does and how to use it, and a `Makefile` providing common commands (`make setup`, `make run`, etc.).

Solutions start in `_experiments/` and graduate to the top level once they are stable and well-documented.

## Usage

Each solution directory has its own `README.md` with specific instructions. In general:

```bash
cd <solution-name>
make setup   # set up the environment (venv, Docker image, etc.)
make run     # run the solution
```

## License

See [LICENSE](LICENSE) if present, or individual solution directories for their own licensing terms.
