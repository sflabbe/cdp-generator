# CDP Generator Repository Structure

This file documents the repository as it exists after the uv packaging migration. It is intentionally limited to the checked-in project structure and does not claim future modules are already implemented.

## Current folder tree

```text
.
├── .gitignore
├── LICENSE
├── MANIFEST.in
├── Makefile
├── README.md
├── REPO_STRUCTURE.md
├── cdp-generator.py                  # Legacy monolithic script kept for compatibility/reference
├── pyproject.toml                    # Package metadata, runtime dependencies, dev dependency group
├── setup.py                          # Minimal compatibility shim, not a dependency source of truth
├── uv.lock                           # pending: generate with `uv lock` in an environment with package index access
├── cdp_generator/
│   ├── __init__.py
│   ├── cli.py                        # Concrete/CDP interactive CLI
│   ├── compression.py
│   ├── core.py
│   ├── export.py
│   ├── material_properties.py
│   ├── plotting.py
│   ├── strain_rate.py
│   ├── temperature.py
│   ├── tension.py
│   └── steel/
│       ├── __init__.py
│       ├── cli.py                    # Steel/Johnson-Cook interactive CLI
│       ├── export.py
│       ├── johnson_cook.py
│       ├── plotting.py
│       └── standards.py
├── docs/
│   └── audits/
│       └── uv_migration_report.md
└── tests/
    ├── __init__.py
    ├── test_basic.py
    └── test_steel.py
```

## Packaging state

- Package type: flat-layout Python package.
- Build backend: `setuptools.build_meta`.
- Minimum Python: `>=3.8` from `pyproject.toml`.
- Runtime dependencies: declared in `project.dependencies`.
- Development dependencies: declared in `[dependency-groups].dev`.
- Legacy dev extra: retained in `[project.optional-dependencies].dev` for compatibility, mirroring the uv dev group.
- Console scripts:
  - `cdp-generator = cdp_generator.cli:main`
  - `cdp-steel = cdp_generator.steel.cli:main`

## Development commands

```bash
uv sync --all-extras --dev
uv run pytest
make lint
make format-check
make lock-check
```

`setup.py` must stay minimal. Dependencies belong in `pyproject.toml`; the lock state belongs in `uv.lock`.
