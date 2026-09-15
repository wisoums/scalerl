# Contributing

ScaleRL uses small, reviewable changes tied to GitHub issues.

## Development workflow

1. Choose or create an issue with clear acceptance criteria.
2. Create a branch from `main` using a prefix such as `feature/`, `fix/`, `experiment/`, `docs/`, or `chore/`.
3. Add or update tests for behavior changes.
4. Run linting, type checks, and tests locally.
5. Open a pull request and link the issue with `Closes #<issue>` when appropriate.
6. Keep research claims in documentation grounded in reproducible experiments.

## Quality gates

Before merge:

```bash
ruff check .
mypy src
pytest
```

For ML experiments, record random seeds, configuration, workload definition, and evaluation metrics. Do not report a single cherry-picked run as a final result.

## Commit style

Prefer concise Conventional-Commit-style messages such as:

- `feat: add replica startup model`
- `test: cover invalid scaling actions`
- `experiment: compare DQN reward weights`
- `docs: document workload assumptions`
