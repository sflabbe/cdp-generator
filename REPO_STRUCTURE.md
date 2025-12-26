# CDP_GENERATOR Repository Structure (with XFEM Integration)

## Complete Folder Tree

```
cdp_generator/
│
├── README.md                          # Main project documentation
├── LICENSE                            # License file
├── CONTRIBUTING.md                    # Contribution guidelines
├── CHANGELOG.md                       # Version history
├── pyproject.toml                     # Modern Python packaging (PEP 517/518)
├── setup.py                           # Backward compatible setup
├── requirements.txt                   # Core dependencies
├── requirements-dev.txt               # Development dependencies
├── .gitignore                         # Git ignore patterns
├── .pre-commit-config.yaml           # Pre-commit hooks
│
├── cdp_generator/                     # Main package
│   ├── __init__.py                   # Package init (version, exports)
│   ├── core.py                       # Core CDP functionality
│   ├── material_properties.py        # EC2/ACI/fib material properties
│   ├── compression.py                # Compression behavior models
│   ├── tension.py                    # Tension behavior models
│   ├── strain_rate.py                # Strain rate effects
│   ├── temperature.py                # Temperature effects
│   ├── export.py                     # Export to Abaqus/LS-DYNA/etc
│   ├── plotting.py                   # Visualization utilities
│   ├── cli.py                        # Command-line interface
│   │
│   └── constitutive/                 # Advanced constitutive models
│       ├── __init__.py
│       ├── drucker_prager.py         # Drucker-Prager plasticity
│       ├── rankine.py                # Rankine (tension cutoff)
│       ├── concrete_damage.py        # CDP damage evolution
│       └── calibration.py            # Parameter calibration tools
│
├── tests/                             # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py                   # Pytest configuration
│   │
│   ├── unit/                         # Unit tests (fast, isolated)
│   │   ├── __init__.py
│   │   ├── test_material_properties.py
│   │   ├── test_compression_curves.py
│   │   ├── test_tension_curves.py
│   │   ├── test_drucker_prager.py
│   │   └── test_calibration.py
│   │
│   ├── integration/                  # Integration tests (components together)
│   │   ├── __init__.py
│   │   ├── test_ec2_workflow.py
│   │   ├── test_export_formats.py
│   │   └── test_constitutive_models.py
│   │
│   └── benchmarks/                   # Performance benchmarks
│       ├── __init__.py
│       ├── bench_dp_stress_update.py
│       └── bench_curve_generation.py
│
├── examples/                          # XFEM and other examples
│   ├── README.md                     # Examples documentation
│   │
│   ├── basic/                        # Basic usage examples
│   │   ├── 01_ec2_material.py
│   │   ├── 02_compression_curve.py
│   │   ├── 03_drucker_prager_calibration.py
│   │   └── 04_export_abaqus.py
│   │
│   ├── xfem/                         # XFEM beam simulation (MAIN FOCUS)
│   │   ├── README.md                # XFEM examples documentation
│   │   ├── requirements.txt         # XFEM-specific dependencies (scipy, matplotlib)
│   │   │
│   │   ├── core/                    # XFEM solver core
│   │   │   ├── __init__.py
│   │   │   ├── xfem_base.py        # Base XFEM formulation
│   │   │   ├── xfem_cohesive.py    # Cohesive crack XFEM
│   │   │   ├── shape_functions.py  # Q4 shape functions, Gauss quadrature
│   │   │   ├── enrichment.py       # Heaviside, tip functions
│   │   │   └── assembly.py         # Global assembly routines
│   │   │
│   │   ├── constitutive/           # Constitutive models (using cdp_generator)
│   │   │   ├── __init__.py
│   │   │   ├── elastic.py          # Linear elastic
│   │   │   ├── cohesive.py         # Cohesive traction-separation
│   │   │   ├── dp_plasticity.py    # Drucker-Prager (uses cdp_generator)
│   │   │   └── hybrid_dp_cohesive.py  # DP bulk + cohesive crack
│   │   │
│   │   ├── benchmarks/             # XFEM benchmark cases
│   │   │   ├── README.md          # Benchmark documentation
│   │   │   ├── beam_3point/       # 3-point bending beam
│   │   │   │   ├── run_beam.py
│   │   │   │   ├── config.yaml     # Benchmark parameters
│   │   │   │   ├── mesh.py         # Mesh generation
│   │   │   │   └── postprocess.py  # Results visualization
│   │   │   │
│   │   │   ├── gutierrez_2004/    # Gutierrez validation case
│   │   │   │   ├── run_gutierrez.py
│   │   │   │   ├── config.yaml
│   │   │   │   └── experimental_data.csv  # Reference data
│   │   │   │
│   │   │   └── double_cantilever/ # DCB test (mode I fracture)
│   │   │       ├── run_dcb.py
│   │   │       └── config.yaml
│   │   │
│   │   ├── profiling/              # Profiling infrastructure
│   │   │   ├── bench_runner.py     # Automated profiling
│   │   │   ├── verify_determinism.py
│   │   │   └── compare_results.py
│   │   │
│   │   ├── tests/                  # XFEM-specific tests
│   │   │   ├── test_shape_functions.py
│   │   │   ├── test_enrichment.py
│   │   │   ├── test_dp_plasticity.py
│   │   │   ├── test_cohesive_law.py
│   │   │   └── test_beam_convergence.py
│   │   │
│   │   └── utils/                  # XFEM utilities
│   │       ├── __init__.py
│   │       ├── mesh_generators.py
│   │       ├── plotting.py
│   │       └── io.py
│   │
│   └── validation/                 # Validation against literature
│       ├── README.md
│       ├── petersson_1981/        # Petersson wedge splitting test
│       ├── hillerborg_1976/       # Hillerborg fictitious crack
│       └── carpinteri_1989/       # Size effect law
│
├── docs/                           # Documentation (Sphinx)
│   ├── conf.py                    # Sphinx configuration
│   ├── index.rst                  # Documentation home
│   ├── Makefile
│   │
│   ├── user_guide/
│   │   ├── installation.rst
│   │   ├── quickstart.rst
│   │   ├── material_models.rst
│   │   └── xfem_tutorial.rst
│   │
│   ├── api_reference/
│   │   ├── cdp_generator.rst
│   │   ├── constitutive.rst
│   │   └── xfem.rst
│   │
│   ├── theory/
│   │   ├── concrete_plasticity.rst
│   │   ├── drucker_prager.rst
│   │   ├── cohesive_fracture.rst
│   │   └── xfem_formulation.rst
│   │
│   └── benchmarks/
│       ├── beam_3point.rst
│       ├── gutierrez_validation.rst
│       └── profiling_results.rst
│
├── scripts/                        # Utility scripts
│   ├── run_all_tests.sh
│   ├── run_benchmarks.sh
│   ├── generate_docs.sh
│   └── profile_xfem.sh
│
├── data/                          # Reference data
│   ├── experimental/              # Experimental datasets
│   │   ├── gutierrez_2004.csv
│   │   ├── petersson_1981.csv
│   │   └── README.md
│   │
│   └── golden/                    # Golden reference results
│       ├── beam_3point_reference.csv
│       └── README.md
│
└── outputs/                       # Generated outputs (gitignored)
    ├── .gitkeep
    └── README.md                  # Explains output structure
```

---

## Key Configuration Files

### `pyproject.toml` (Modern Python Packaging)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cdp_generator"
version = "2.0.0"
description = "Concrete Damaged Plasticity parameter generator with XFEM capabilities"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["concrete", "plasticity", "CDP", "XFEM", "finite-element"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "matplotlib>=3.3.0",
]

[project.optional-dependencies]
xfem = [
    "scipy>=1.7.0",
    "matplotlib>=3.3.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "pytest-benchmark>=4.0.0",
    "black>=22.0.0",
    "ruff>=0.0.250",
    "mypy>=0.990",
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
]
docs = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "myst-parser>=0.18.0",
]

[project.scripts]
cdp = "cdp_generator.cli:main"

[project.urls]
Homepage = "https://github.com/yourusername/cdp_generator"
Documentation = "https://cdp-generator.readthedocs.io"
Repository = "https://github.com/yourusername/cdp_generator"
"Bug Tracker" = "https://github.com/yourusername/cdp_generator/issues"

[tool.setuptools.packages.find]
where = ["."]
include = ["cdp_generator*"]
exclude = ["tests*", "examples*", "docs*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --cov=cdp_generator --cov-report=html --cov-report=term"

[tool.black]
line-length = 100
target-version = ["py38", "py39", "py310", "py311"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

---

### `.gitignore`

```gitignore
# Byte-compiled / optimized
__pycache__/
*.py[cod]
*$py.class
*.so

# Distribution / packaging
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.tox/
.coverage
.coverage.*
htmlcov/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Profiling
prof.out
*.prof
pstats_*.txt
*.pstats

# XFEM outputs
outputs/
results.csv
*.png
*.pdf
!docs/**/*.png
!data/experimental/*.png
!golden/**/*.png

# Virtual environments
.venv/
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Documentation builds
docs/_build/
docs/_static/
docs/_templates/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Data (large files)
*.h5
*.hdf5
*.vtk
*.vtu
```

---

### `tests/conftest.py` (Pytest Configuration)

```python
"""Pytest configuration for cdp_generator tests."""

import pytest
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Define test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

@pytest.fixture
def c20_25_props():
    """C20/25 concrete properties for testing."""
    from cdp_generator import material_properties as mp
    return mp.ec2_concrete("C20/25")

@pytest.fixture
def dp_params():
    """Default Drucker-Prager parameters."""
    return {
        "phi_deg": 36.0,
        "alpha": 0.4242,  # tan(36°)/√3
        "E": 26.171e9,    # Pa
        "nu": 0.2,
    }

@pytest.fixture(scope="session")
def golden_data_dir():
    """Golden reference data directory."""
    return Path(__file__).parent.parent / "data" / "golden"

@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    """Temporary output directory for test results."""
    return tmp_path_factory.mktemp("test_outputs")
```

---

### `examples/xfem/benchmarks/beam_3point/config.yaml`

```yaml
# 3-Point Bending Beam Benchmark Configuration

benchmark:
  name: "beam_3point_dp_cohesive"
  description: "RC beam in 3-point bending with DP plasticity and cohesive crack"
  date: "2025-12-26"

geometry:
  L: 5.0          # m (span)
  H: 0.5          # m (height)
  b: 0.25         # m (width)
  cover: 0.05     # m (concrete cover)

material:
  concrete:
    grade: "C20/25"
    model: "drucker_prager"  # Options: elastic, drucker_prager, cdp
    dp_friction_angle: 36.0  # degrees

  steel:
    n_bars: 2
    diameter: 0.012  # m (12mm)
    E: 200.0e9      # Pa
    fy: 500.0e6     # Pa
    fu: 540.0e6     # Pa

mesh:
  nx: 120
  ny: 18
  element_type: "Q4"

loading:
  type: "displacement_control"
  umax: 0.010     # m (10mm)
  nsteps: 30

crack:
  model: "cohesive_xfem"
  crack_rho: 0.25         # m (nonlocal radius)
  crack_margin: 0.30      # m (from supports)
  crack_stop_y: 0.25      # m (arrest height)
  tip_enr_radius: 0.0     # m (0 = Heaviside only)
  Kn_factor: 0.1          # Cohesive penalty scaling

solver:
  newton_maxit: 60
  newton_tol_r: 1.0e-2
  newton_tol_rel: 1.0e-6
  line_search: true
  max_subdiv: 14

output:
  save_results: true
  save_plots: true
  save_stress_fields: true
  save_vtk: false
```

---

### `examples/xfem/profiling/bench_runner.py`

```python
#!/usr/bin/env python3
"""
XFEM Benchmark Runner with Profiling and Manifest Generation.

Usage:
    python bench_runner.py --config benchmarks/beam_3point/config.yaml \
                          --tag beam_dp_cohesive
"""

import os
import sys
import subprocess
import argparse
import json
import time
import platform
from pathlib import Path
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def enforce_deterministic_env():
    """Set environment for single-threaded deterministic execution."""
    env_vars = {
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'NUMBA_NUM_THREADS': '1',
    }
    for key, val in env_vars.items():
        os.environ[key] = val
        print(f"[env] {key}={val}")

def get_git_info():
    """Get git commit and dirty status."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                        stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'],
                                        stderr=subprocess.DEVNULL).decode().strip()
        dirty = len(status) > 0
        return commit, dirty
    except:
        return "UNKNOWN", False

def get_system_info():
    """Get system information."""
    import numpy as np
    import scipy

    cpu_count = os.cpu_count()
    if platform.system() == 'Linux':
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_model = line.split(':')[1].strip()
                        break
        except:
            cpu_model = "Unknown"
    else:
        cpu_model = platform.processor()

    return {
        'python_version': sys.version,
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'os': platform.system(),
        'os_release': platform.release(),
        'cpu_model': cpu_model,
        'cpu_count': cpu_count,
    }

def run_benchmark(config_file, tag):
    """Run XFEM benchmark with profiling."""

    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directories
    output_root = Path('outputs') / tag
    prof_dir = Path('outputs') / 'profiling' / tag
    output_root.mkdir(parents=True, exist_ok=True)
    prof_dir.mkdir(parents=True, exist_ok=True)

    # Enforce deterministic environment
    enforce_deterministic_env()

    # Prepare profiling command
    prof_file = prof_dir / 'prof.out'
    benchmark_script = Path(config_file).parent / 'run_beam.py'

    cmd = [
        sys.executable, '-m', 'cProfile', '-o', str(prof_file),
        str(benchmark_script),
        '--config', str(config_file),
        '--output', str(output_root),
        '--no-plots'  # Disable plotting for profiling
    ]

    print(f"\n{'='*70}")
    print(f"XFEM BENCHMARK: {tag}")
    print(f"Config: {config_file}")
    print(f"{'='*70}\n")
    print(f"Command: {' '.join(cmd)}\n")

    # Run with profiling
    t0 = time.time()
    result = subprocess.run(cmd, env=os.environ.copy())
    wall_time = time.time() - t0

    if result.returncode != 0:
        print(f"\n[ERROR] Benchmark failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n[profiling] Wall time: {wall_time:.2f} s")

    # Generate pstats reports
    import pstats

    cumtime_file = prof_dir / 'pstats_cumtime.txt'
    with open(cumtime_file, 'w') as f:
        s = pstats.Stats(str(prof_file), stream=f)
        s.strip_dirs()
        s.sort_stats('cumulative')
        s.print_stats(60)

    tottime_file = prof_dir / 'pstats_tottime.txt'
    with open(tottime_file, 'w') as f:
        s = pstats.Stats(str(prof_file), stream=f)
        s.strip_dirs()
        s.sort_stats('tottime')
        s.print_stats(60)

    # Generate manifest
    commit, dirty = get_git_info()
    sys_info = get_system_info()

    manifest = {
        'tag': tag,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'git': {'commit': commit, 'dirty': dirty},
        'command': ' '.join(cmd),
        'environment': sys_info,
        'config': config,
        'wall_time_seconds': wall_time,
    }

    manifest_file = prof_dir / 'manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*70}")
    print(f"PROFILING COMPLETE")
    print(f"Results: {output_root}")
    print(f"Profile: {prof_dir}")
    print(f"{'='*70}\n")

    return wall_time

def main():
    parser = argparse.ArgumentParser(description='XFEM Benchmark Runner')
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    parser.add_argument('--tag', required=True, help='Benchmark tag')
    args = parser.parse_args()

    run_benchmark(args.config, args.tag)

if __name__ == '__main__':
    main()
```

---

### `scripts/run_all_tests.sh`

```bash
#!/bin/bash
# Run all tests with coverage

set -e

echo "=========================================="
echo "Running CDP Generator Test Suite"
echo "=========================================="

# Unit tests
echo ""
echo ">>> Running unit tests..."
pytest tests/unit/ -v --cov=cdp_generator --cov-report=html --cov-report=term

# Integration tests
echo ""
echo ">>> Running integration tests..."
pytest tests/integration/ -v

# XFEM tests
echo ""
echo ">>> Running XFEM tests..."
pytest examples/xfem/tests/ -v

# Benchmarks (fast mode)
echo ""
echo ">>> Running performance benchmarks..."
pytest tests/benchmarks/ -v --benchmark-only --benchmark-disable-gc

echo ""
echo "=========================================="
echo "✓ ALL TESTS PASSED"
echo "Coverage report: htmlcov/index.html"
echo "=========================================="
```

---

### `examples/xfem/README.md`

```markdown
# XFEM Examples for CDP Generator

This directory contains Extended Finite Element Method (XFEM) implementations
using constitutive models from cdp_generator.

## Structure

```
xfem/
├── core/              # XFEM solver implementation
├── constitutive/      # Constitutive models (DP, cohesive, hybrid)
├── benchmarks/        # Benchmark cases
├── profiling/         # Profiling infrastructure
├── tests/             # XFEM-specific tests
└── utils/             # Utilities (mesh, plotting, I/O)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[xfem]"
```

### 2. Run 3-Point Bending Benchmark

```bash
cd benchmarks/beam_3point
python run_beam.py --config config.yaml
```

### 3. Profile the Simulation

```bash
cd ../../profiling
python bench_runner.py --config ../benchmarks/beam_3point/config.yaml \
                      --tag beam_baseline
```

### 4. Run Tests

```bash
pytest tests/ -v
```

## Benchmarks

### Beam 3-Point Bending
- **Location:** `benchmarks/beam_3point/`
- **Description:** RC beam with DP plasticity + cohesive crack
- **Reference:** Validates against linear elastic baseline

### Gutierrez 2004
- **Location:** `benchmarks/gutierrez_2004/`
- **Description:** Validation against experimental data
- **Reference:** Gutierrez (2004) PhD thesis

### Double Cantilever Beam (DCB)
- **Location:** `benchmarks/double_cantilever/`
- **Description:** Mode I fracture test
- **Reference:** Standard ASTM test

## Constitutive Models

### Available Models

1. **Elastic** (`constitutive/elastic.py`)
   - Linear elastic (baseline)

2. **Drucker-Prager** (`constitutive/dp_plasticity.py`)
   - Pressure-dependent plasticity
   - Calibrated from cdp_generator EC2 curves

3. **Cohesive** (`constitutive/cohesive.py`)
   - Traction-separation law
   - Bilinear softening

4. **Hybrid DP+Cohesive** (`constitutive/hybrid_dp_cohesive.py`)
   - DP for bulk compression
   - Cohesive for discrete crack

## Profiling

All benchmarks support profiling via `bench_runner.py`:

```bash
# Run with profiling
python profiling/bench_runner.py \
    --config benchmarks/beam_3point/config.yaml \
    --tag my_benchmark

# Results in:
outputs/
├── my_benchmark/
│   └── results.csv
└── profiling/
    └── my_benchmark/
        ├── prof.out
        ├── pstats_cumtime.txt
        ├── pstats_tottime.txt
        └── manifest.json
```

## Testing

```bash
# All XFEM tests
pytest tests/ -v

# Specific test
pytest tests/test_dp_plasticity.py::test_uniaxial_compression -v

# With coverage
pytest tests/ --cov=constitutive --cov=core
```

## Documentation

See `../../docs/user_guide/xfem_tutorial.rst` for detailed tutorial.
```

---

## Summary

This structure provides:

✅ **Proper Python package** (pyproject.toml, setup.py)
✅ **Comprehensive testing** (unit, integration, benchmarks)
✅ **XFEM integration** (as examples, not core package)
✅ **Profiling infrastructure** (bench_runner, determinism checks)
✅ **Documentation** (Sphinx, user guide, API reference)
✅ **Validation cases** (Gutierrez, Petersson, etc.)
✅ **CI/CD ready** (.pre-commit, pytest, coverage)
✅ **HPC discipline** (reproducibility, profiling, golden data)

The key insight: **XFEM is in `examples/`, not `cdp_generator/`** because:
- cdp_generator = material models (core functionality)
- XFEM = application using cdp_generator (example usage)
- Keeps package scope focused
- Allows independent testing/profiling
