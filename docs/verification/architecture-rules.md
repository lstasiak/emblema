# Architecture rules turn CI red on a deliberate violation

Purpose: show that every import-linter contract in `pyproject.toml` breaks for the reason it exists,
not merely passes because the code is still small. The same contracts run in CI (job
`architecture rules`, one step per rule family) and in `tests/architecture/test_import_contracts.py`.

## 2026-09-08 — import-linter 2.15, Python 3.14.5, Windows (local)

Method: each negative test copies `src/emblema` into a temporary directory, adds one violating
module, puts the copy first on `PYTHONPATH` so it shadows the editable install, and runs
`lint-imports --config pyproject.toml --no-cache` with the **unchanged** contracts. The report must
name the contract as `BROKEN` and quote the injected import. The positive tests run the same command
against the real package (0 broken) and against a copy where a context imports another context's
`contracts` package (still 0 broken: the exemption works).

| Contract id | Injected import | Report excerpt | Exit |
|---|---|---|---|
| `context-independence` | `emblema.catalog.domain.leak -> emblema.pretraining.domain` | `emblema.catalog is not allowed to import emblema.pretraining` | 1 |
| `hexagonal-layers` | `emblema.catalog.domain.entity -> emblema.catalog.adapters.persistence` | `emblema.catalog.domain is not allowed to import emblema.catalog.adapters` | 1 |
| `hexagonal-layers` | `emblema.catalog.application.use_case -> emblema.catalog.adapters.persistence` | `emblema.catalog.application is not allowed to import emblema.catalog.adapters` | 1 |
| `pure-core` | `emblema.catalog.domain.model -> torch` | `emblema.catalog.domain is not allowed to import torch` | 1 |
| `pure-core` | `emblema.catalog.application.dto -> pydantic` | `emblema.catalog.application is not allowed to import pydantic` | 1 |
| `pure-core` | `emblema.shared.kernel.tensor -> numpy` | `emblema.shared is not allowed to import numpy` | 1 |
| `config-only-at-the-edges` | `emblema.catalog.application.use_case -> emblema.config.settings` | `emblema.catalog.application is not allowed to import emblema.config` — and `pure-core` breaks too, through the chain `config.settings -> pydantic` (indirect imports are forbidden as well) | 1 |
| `contracts-depend-only-on-shared` | `emblema.catalog.contracts.published -> emblema.catalog.domain` | `emblema.catalog.contracts is not allowed to import emblema.catalog.domain` | 1 |
| `shared-imports-no-context` | `emblema.shared.kernel.leak -> emblema.catalog` | `emblema.shared is not allowed to import emblema.catalog` | 1 |
| `entrypoints-are-outermost` | `emblema.catalog.adapters.cli_glue -> emblema.entrypoints` | `emblema.catalog is not allowed to import emblema.entrypoints` | 1 |

Positive runs: real package `Contracts: 7 kept, 0 broken.` (exit 0, two warnings: the `contracts`
exemption matches nothing until the first context publishes one); copy with
`emblema.catalog.application.use_case -> emblema.pretraining.contracts.published` → 0 broken.

Behaviour of the tool confirmed on this version, relied on by the configuration:

- a layer wrapped in parentheses may be absent from a container (all four contexts have no layers yet);
- a wildcard `source_modules` entry that matches nothing is accepted; an explicit non-existent
  source module is an error (`Module 'x' does not exist.`) — hence `emblema.shared` as a whole,
  not `emblema.shared.ports` / `emblema.shared.events`;
- an explicit non-existent `forbidden_modules` entry is accepted (`emblema.entrypoints`, frameworks
  not installed);
- `--contract <id>` limits a run to the given contracts and may be repeated;
- unmatched `ignore_imports` produce a warning, not a failure, with `unmatched_ignore_imports_alerting = "warn"`.

Coverage gate: `uv run pytest --cov` → `Required test coverage of 92.0% reached. Total coverage:
100.00%` (`fail_under` read from `[tool.coverage.report]`, no CLI flag needed). Runtime: architecture
tests 10 s, whole suite 15 s on this machine.

CI run on GitHub Actions: PR #2, commit `2af9337` — <https://github.com/lstasiak/emblema/actions/runs/34167005105>:
jobs `lint`, `types`, `architecture rules` (4 steps), `test (3.12)`, `test (3.14)` all green.
Coverage visibility (job summary, PR comment, badge data branch) was added afterwards; its first run
is the next push on this PR, the badge data branch appears with the first merge to `main`.

## 2026-09-08 — after merge of PR #2 (`a7e2e71`)

- CI on `main`: <https://github.com/lstasiak/emblema/actions/runs/34168630829> — all jobs green.
- Coverage comment posted on PR #2 by the `test (3.14)` job (same-repository PR, write token);
  the `Post coverage comment` workflow correctly skipped on the `main` push (PR events only).
- Badge data branch `python-coverage-comment-action-data` created with `endpoint.json`, `badge.svg`,
  `data.json`. The README badge resolves once the repository is public (shields fetches anonymously).

## 2026-09-08 — the first `contracts` package, `shared.events`, and the contract they needed

import-linter 2.15, Python 3.14.5, Windows (local). Same method as above.

| Contract id | Injected import | Exit |
|---|---|---|
| `shared-layers` | `emblema.shared.events.leak -> emblema.shared.ports.clock` | 1 |
| `shared-layers` | `emblema.shared.kernel.leak -> emblema.shared.events.domain_event` | 1 |
| `domain-shares-only-identity-with-contracts` | `emblema.catalog.domain.leak -> emblema.catalog.contracts.corpus_version_ref` | 1 |

Positive runs: real package `Contracts: 9 kept, 0 broken.` (no warnings); copy where a domain module
imports `emblema.catalog.contracts.identifiers` → 0 broken (the exemption works).

Tool behaviour confirmed, relied on by the configuration:

- `**` in `ignore_imports` matches **one or more** segments: `emblema.** -> emblema.*.contracts.**`
  does not exempt `import emblema.pretraining.contracts` (bare package). Checked with a throwaway
  config: the bare import breaks independence under the `**` pattern alone and is exempted only by
  a separate `emblema.** -> emblema.*.contracts` pattern. That pattern matched nothing in the real
  package and unmatched ignores fail the run by default (`unmatched_ignore_imports_alerting`
  defaults to `error`), so it was dropped: consumers import contract modules, never the bare package.
- `ignore_imports` on a `forbidden` contract may use wildcards on the importer side
  (`emblema.catalog.domain.* -> emblema.catalog.contracts.identifiers`).
- A contract name longer than ~72 characters is wrapped in the report, which breaks the
  `<name> BROKEN` assertion of the negative tests; names are kept short.

Run-time isolation: `tests/architecture/test_context_isolation.py` starts a fresh interpreter,
imports `emblema.catalog.contracts.events` (and a test module standing in for a downstream
context) and lists `sys.modules`: no `emblema.catalog.domain`, `.application` or `.adapters`.
