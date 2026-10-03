# CDP Generator

A comprehensive Python package for generating material model parameters for ABAQUS finite element simulations:

1. **Concrete Damage Plasticity (CDP)** - For concrete materials
2. **Johnson-Cook (JC) Plasticity** - For steel materials

Both models support strain-rate and temperature-dependent properties.

## Features

### Concrete (CDP)
- **Authority-split facade (G0)**: `cdp_generator.concrete` separates physical concrete data and provenance from constitutive-model parameters while preserving the legacy API
- **Legacy compatibility profile**: `legacy_v1` reproduces current behavior without claiming verified fib/Eurocode authority
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Strain Rate Dependent Analysis**: Calculates CDP parameters for multiple strain rates
- **Temperature Dependent Analysis**: Computes temperature-dependent properties based on Eurocode
- **Multiple Models**: Supports both bilinear and power law tension softening models
- **Comprehensive Output**: Generates stress-strain curves, damage parameters, and material properties

### Concrete authority facade

G0 adds a parallel, backward-compatible API without moving the existing concrete modules:

```python
from cdp_generator.concrete import Concrete

concrete = Concrete.from_mean_strength(
    f_cm=38.0, e_c1=0.0022, e_clim=0.0035, profile="legacy_v1"
)
physical = concrete.physical
abaqus = concrete.to_abaqus_cdp(calibration="abaqus_cdp_legacy")
```

`legacy_v1` is a compatibility authority, not a claim of verified normative-code equivalence. Verified fib/EC2 profiles and the Grassl CDPM2 backend are reserved for later gates. See `docs/concrete_authority_architecture.md`.

### Steel (Johnson-Cook)
- **Standards Database**: Built-in properties for EC2 (Eurocode), ACI/ASTM, and NCh standards
- **Automatic Calibration**: Generates Johnson-Cook parameters from standard steel specifications
- **Custom Materials**: Full support for user-defined steel properties
- **Multi-Rate/Temperature**: Analyzes behavior across multiple strain rates and temperatures
- **ABAQUS Export**: Optional material card export for ABAQUS (experimental)

### Common Features
- **Easy Integration**: Can be imported and used in other Python projects
- **Excel Export**: Automatically exports results to Excel for use in ABAQUS
- **Visualization**: Built-in plotting functions for all results

## Development setup

This repository uses **uv** as the maintained dependency, environment and command runner. The source of truth is:

- `pyproject.toml` for package metadata, Python 3.13 policy, and dependency groups
- `.python-version` to pin the local uv interpreter family to Python 3.13
- `uv.lock` for the resolved environment
- `Makefile` for repeatable development commands

### Install uv

Install uv using the official Astral instructions for your platform. A common Unix installer is:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create or synchronize the environment

The maintained baseline is Python 3.13.

```bash
uv python install 3.13
uv sync --all-extras --dev
```

Equivalent make target:

```bash
make sync
```

### Run tests

```bash
uv run pytest
# or
make test
```

### Run lint, type checking, and formatting tools

Development targets Python 3.13. Ruff is the single linter/formatter/import sorter, and mypy checks the maintained package API.

```bash
make lint
make typecheck
make format-check
make format
make check
```

### Run the interactive CLIs

```bash
uv run cdp-generator
uv run cdp-steel
```

Equivalent make targets:

```bash
make run-cdp
make run-steel
```

### Update the lockfile

```bash
uv lock
make lock
```

Check whether the lockfile is current:

```bash
uv lock --check
make lock-check
```

### Add dependencies

Runtime dependency:

```bash
uv add <package>
```

Development dependency:

```bash
uv add --dev <package>
```

### Legacy packaging policy

`setup.py` is retained only as a minimal compatibility shim for legacy build frontends. Do not add dependencies or project metadata there.

No `requirements.txt` files are present in this repository after the migration. If a compatibility export is needed for an external deployment target, generate it from uv and treat it as an artifact, not as the source of truth.

### User installation outside development

For normal package use from another uv-based project, prefer adding the package as a dependency from the published package or Git source once available. Development of this repository itself should use the commands above.

## Usage

### Command Line Interface

#### Concrete (CDP)

Run the interactive CDP CLI:

```bash
cdp-generator
```

Or if installed in development mode:

```bash
python -m cdp_generator.cli
```

#### Steel (Johnson-Cook)

Run the interactive steel CLI:

```bash
cdp-steel
```

Or if installed in development mode:

```bash
python -m cdp_generator.steel.cli
```

The CLI will guide you through:
1. Selecting a steel standard (EC2/ACI/NCh) or custom properties
2. Choosing a grade and optional overrides
3. Calibrating Johnson-Cook parameters
4. Specifying strain rates and temperatures for analysis
5. Generating curves and exporting results

### As a Python Library

#### Strain Rate Dependent Analysis

```python
from cdp_generator import calculate_stress_strain, export_to_excel, plot_all_results

# Define input parameters
f_cm = 28.0              # Mean compressive strength [MPa]
e_c1 = 0.0022            # Strain at peak compressive strength [-]
e_clim = 0.0035          # Ultimate strain [-]
l_ch = 1.0               # Characteristic element length [mm]
strain_rates = [0, 2, 30, 100]  # Strain rates [1/s]

# Calculate stress-strain relationships
results = calculate_stress_strain(f_cm, e_c1, e_clim, l_ch, strain_rates)

# Access results
print(f"Tensile strength: {results['properties']['tensile strength']:.2f} MPa")
print(f"Dilation angle: {results['properties']['dilation angle']:.2f}°")
print(f"CDP Kc: {results['properties']['Kc']:.2f}")
print(f"CDP fb/fc: {results['properties']['fbfc']:.2f}")

# Plot results
plot_all_results(results, strain_rates, mode='strain_rate')

# Export to Excel
export_to_excel(results, strain_rates, mode='strain_rate', filename='CDP-Results.xlsx')
```

#### Temperature Dependent Analysis

```python
from cdp_generator import calculate_stress_strain_temp, export_to_excel
import numpy as np

# Define input parameters
f_cm = 28.0
e_c1 = 0.0022
e_clim = 0.0035
l_ch = 1.0

# Calculate temperature-dependent properties
results = calculate_stress_strain_temp(f_cm, e_c1, e_clim, l_ch, verbose=True)

# Eurocode temperatures are used by default
temperatures = np.array([20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100])

# Export results
export_to_excel(results, temperatures, mode='temperature', filename='CDP-Results-Temp.xlsx')
```

#### Using Individual Functions (CDP)

```python
from cdp_generator import (
    calculate_concrete_strength_properties,
    calculate_elastic_modulus,
    calculate_cdp_parameters,
    calculate_compression_behavior,
    calculate_tension_bilinear
)

# Calculate material properties
f_cm = 28.0
strength_props = calculate_concrete_strength_properties(f_cm)
print(f"f_ck: {strength_props['f_ck']:.2f} MPa")
print(f"f_ctm: {strength_props['f_ctm']:.2f} MPa")

elastic_props = calculate_elastic_modulus(f_cm)
print(f"E_ci: {elastic_props['E_ci']:.2f} MPa")
print(f"E_c: {elastic_props['E_c']:.2f} MPa")
```

---

### Steel (Johnson-Cook) Library Usage

#### Basic Usage - From Standard Specification

```python
from cdp_generator.steel import (
    get_steel_spec,
    calibrate_jc_from_spec,
    generate_jc_curves_multicase,
    export_steel_to_excel,
    plot_steel_results
)

# 1. Get steel specification from standard
spec = get_steel_spec("EC2", "B500C")  # Eurocode B500 Class C

# 2. Calibrate Johnson-Cook parameters
params = calibrate_jc_from_spec(spec, verbose=True)

# 3. Generate stress-strain curves for multiple cases
results = generate_jc_curves_multicase(
    params=params,
    E=200000,  # Elastic modulus [MPa]
    strain_rates=[1e-4, 1e-3, 1e-2, 1, 10, 100],  # [1/s]
    temperatures=[20, 200, 400, 600, 800],  # [°C]
    eps_max=0.20,  # Maximum strain
    n_points=100,
    output_kind="true"  # "true" or "engineering"
)

# 4. Export to Excel
export_steel_to_excel(results, filename="Steel-B500C-Results.xlsx")

# 5. Plot results
plot_steel_results(results, show=True)
```

#### Available Standards and Grades

```python
from cdp_generator.steel import list_available_standards, print_standards_info

# List all available standards
standards = list_available_standards()
print(standards)
# {'EC2': ['B500A', 'B500B', 'B500C'],
#  'ACI': ['A615_60', 'A615_75', 'A706_60'],
#  'NCh': ['A630-420H', 'A440-280H']}

# Print detailed information about all standards
print_standards_info()
```

#### Custom Steel Properties

```python
from cdp_generator.steel import get_steel_spec, calibrate_jc_from_spec

# Define custom steel with your own properties
spec = get_steel_spec(
    standard="Custom",
    grade="MyCustomSteel",
    overrides={
        "fy": 550,      # Yield strength [MPa]
        "fu": 700,      # Ultimate strength [MPa]
        "Agt": 8.5,     # Elongation at max force [%]
        "E": 200000,    # Elastic modulus [MPa]
    }
)

# Calibrate JC parameters
params = calibrate_jc_from_spec(spec, verbose=True)
```

#### Override Standard Values

```python
from cdp_generator.steel import get_steel_spec

# Use standard as base, but override specific values
spec = get_steel_spec(
    standard="EC2",
    grade="B500C",
    overrides={
        "fy": 550,  # Override yield strength
        "Agt": 9.0  # Override elongation
    }
)
```

#### Generate Single Curve

```python
from cdp_generator.steel import generate_jc_curve, plot_single_curve

# Generate a single stress-strain curve
curve = generate_jc_curve(
    params=params,
    E=200000,
    eps_max=0.15,
    n_points=100,
    epsdot=1e-3,  # Quasi-static
    T=20,  # Room temperature
    output_kind="true"
)

# Access curve data
strain = curve['strain']
stress = curve['stress']
plastic_strain = curve['plastic_strain']

# Plot single curve
plot_single_curve(curve, show=True)
```

#### Advanced: Manual Johnson-Cook Parameters

```python
from cdp_generator.steel import JohnsonCookParams, johnson_cook_flow_stress
import numpy as np

# Define JC parameters manually
params = JohnsonCookParams(
    A=500,      # Yield stress [MPa]
    B=320,      # Hardening coefficient [MPa]
    n=0.28,     # Hardening exponent
    C=0.014,    # Strain rate coefficient
    m=1.06,     # Thermal softening exponent
    epsdot0=1e-3,  # Reference strain rate [1/s]
    T_room=20,     # Room temperature [°C]
    T_melt=1500    # Melting temperature [°C]
)

# Calculate flow stress at specific conditions
eps_p = np.linspace(0, 0.20, 50)  # Plastic strain
sigma = johnson_cook_flow_stress(
    eps_p=eps_p,
    epsdot=100,  # High strain rate
    T=400,       # Elevated temperature
    params=params
)
```

#### Export ABAQUS Material Card (Experimental)

```python
from cdp_generator.steel import export_abaqus_material_card

# Export ABAQUS input format
export_abaqus_material_card(
    params=params,
    E=200000,
    nu=0.30,
    filename="steel_material.inp"
)

# ⚠️ IMPORTANT: Always verify the output against ABAQUS documentation
#    for your version. This is a basic template.
```

#### Compare Multiple Standards

```python
from cdp_generator.steel import compare_standards

# Generate results for multiple standards
results_list = []
labels = []

for standard, grade in [("EC2", "B500C"), ("ACI", "A615_60"), ("NCh", "A630-420H")]:
    spec = get_steel_spec(standard, grade)
    params = calibrate_jc_from_spec(spec)
    results = generate_jc_curves_multicase(params, E=200000, strain_rates=[1e-3])
    results_list.append(results)
    labels.append(f"{standard} {grade}")

# Plot comparison
compare_standards(results_list, labels, show=True)
```

---

### ⚠️ IMPORTANT NOTES FOR STEEL MODULE

**Standards Database Disclaimer:**

The built-in standard values are **APPROXIMATE** and provided as convenient defaults. They are based on typical minimum requirements but may not reflect:
- The latest version of the standard
- Regional variations
- Specific manufacturer specifications
- Diameter-dependent variations

**Always verify critical values with the actual standard for your jurisdiction and application.**

You can override any value:
```python
spec = get_steel_spec("EC2", "B500C", overrides={"fy": 550, "fu": 650, "Agt": 8.0})
```

**Johnson-Cook Calibration:**

The calibration from standard specifications makes these assumptions:
- Quasi-static loading at room temperature for base calibration
- Strain hardening exponent `n` is estimated from ductility class (can be overridden)
- Rate sensitivity `C` and thermal softening `m` default to 0 (can be specified)

For accurate results:
- Use experimental stress-strain data when available
- Validate calibrated parameters against test data
- Consider performing sensitivity analyses

---

### Integration with Other Projects

For use in other repositories (e.g., a main ABAQUS materials library):

```python
# In your main materials repository
from cdp_generator import calculate_stress_strain
import json

def generate_abaqus_material_card(concrete_grade):
    """Generate ABAQUS material input for a concrete grade."""

    # Define properties based on grade
    if concrete_grade == "C20/25":
        f_cm = 28.0
        e_c1 = 0.0022
    elif concrete_grade == "C30/37":
        f_cm = 38.0
        e_c1 = 0.0023
    # Add more grades...

    # Calculate CDP parameters
    results = calculate_stress_strain(
        f_cm=f_cm,
        e_c1=e_c1,
        e_clim=0.0035,
        l_ch=1.0,
        strain_rates=[0]
    )

    # Extract parameters for ABAQUS input
    props = results['properties']

    return {
        'name': f'Concrete_{concrete_grade}',
        'elasticity': props['elasticity'],
        'poisson': props['poisson'],
        'dilation_angle': props['dilation angle'],
        'eccentricity': 0.1,  # Default value
        'fb0_fc0': props['fbfc'],
        'K': props['Kc'],
        'viscosity': 0.0
    }
```

## Module Structure

```
cdp_generator/
├── __init__.py              # Public API exports
│
├── # === CONCRETE (CDP) MODULES ===
├── core.py                  # Main CDP calculation functions
├── material_properties.py   # Basic material property calculations
├── strain_rate.py           # Strain rate effect functions
├── temperature.py           # Temperature effect functions
├── compression.py           # Compression behavior (CEB-90 model)
├── tension.py               # Tension behavior (bilinear, power law)
├── plotting.py              # Visualization functions
├── export.py                # Excel export functions
├── cli.py                   # CDP command-line interface
│
└── # === STEEL (JOHNSON-COOK) MODULES ===
    steel/
    ├── __init__.py          # Steel subpackage exports
    ├── johnson_cook.py      # JC model core & dataclasses
    ├── standards.py         # Standards database & calibration
    ├── export.py            # Steel Excel export
    ├── plotting.py          # Steel visualization
    └── cli.py               # Steel command-line interface
```

## Input Parameters

| Parameter | Description | Units | Default |
|-----------|-------------|-------|---------|
| `f_cm` | Mean compressive strength | MPa | 28 |
| `e_c1` | Strain at peak compressive strength | - | 0.0022 |
| `e_clim` | Ultimate strain | - | 0.0035 |
| `l_ch` | Characteristic element length | mm | 1 |
| `strain_rates` | List of strain rates (strain rate mode) | 1/s | [0, 2, 30, 100] |

## Output

The package provides:

1. **Material Properties**:
   - Elastic modulus
   - Poisson's ratio
   - Tensile strength
   - Fracture energy
   - CDP parameters (dilation angle, Kc, fb/fc)

2. **Stress-Strain Data**:
   - Compression stress-strain curves
   - Compression inelastic strain
   - Compression damage
   - Tension crack opening curves
   - Tension cracking strain
   - Tension damage

3. **Visualization**:
   - Multiple stress-strain plots
   - Damage evolution curves

4. **Excel Export**:
   - Ready-to-use data for ABAQUS input

## Theory and References

This package implements:

- **Material Properties**: Based on Eurocode 2, fib Model Code 2010
- **Compression Behavior**: CEB-90 model
- **Tension Behavior**: Bilinear and power law models (FIB2010)
- **Strain Rate Effects**: Dynamic Increase Factors (DIF)
- **Temperature Effects**: Eurocode temperature-dependent reduction factors

## Disclaimer

⚠️ This package is provided for research purposes. Users should:

- Verify all outputs for plausibility
- Understand the underlying assumptions and models
- Use appropriate safety factors for design
- Validate results against experimental data when possible

The authors accept no liability for the use of this software in personal, academic, or commercial applications.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License - see LICENSE file for details

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.


## Legacy concrete web UI

Install the optional interactive stack:

```bash
pip install "cdp-generator[web]"
# Repository development:
uv sync --locked --extra web
```

Run from the repository root:

```bash
streamlit run cdp_generator/web/app.py
# Or: uv run --locked --extra web streamlit run cdp_generator/web/app.py
```

Choose strain rate or temperature, enter the legacy inputs, and press Calculate.
The last successful result remains visible with its input summary. Inspect nine
interactive plots, properties and raw curves; download canonical JSON or the
backward-compatible eight-sheet XLSX in the browser. The additive **Abaqus CDP**
tab builds a static-reference full legacy material card with backend validation,
focused export curves, provenance JSON and deterministic `.inp` download.

The **Legacy curves** workflow exposes the historical concrete CDP curve
generator without claiming EC2/fib curve qualification. Temperature cases come
from the kernel; damage uses the first case. Authority-aware physical definitions
and CDPM2 conversion are available through a separate workflow. See [web architecture](docs/web_architecture.md).

### Windows qualification

Run the complete qualification in UTF-8 mode with a fail-fast PowerShell script:

```powershell
.\scripts\qualify_web.ps1
```

`.gitattributes` pins scientific text artifacts to LF. For an existing Windows
checkout, the script first restores CRLF-converted frozen files **only if** their
LF bytes exactly match the original recorded SHA256. Other content changes are
rejected before any file is written. No frozen hash or baseline is regenerated.
Frozen tests retain their original bytes; `PYTHONUTF8=1` supplies the portable
encoding and is restored afterwards. Execute the `.ps1` as a file: pasting
individual commands into an interactive shell can continue after a `throw`.


## Authority-aware material / CDPM2 (WEB-M2)

Run the same Streamlit command above and select **Authority-aware material / CDPM2**.
Choose fib MC2010, EC2 2004 or EC2 2023, a registry class, and profile inputs.
Press **Build material / Assess CDPM2** to inspect physical values, resolution,
full provenance, requested/effective configuration, and domain readiness.

- fib `C30` is READY with the static Grassl 2013 calibration.
- EC2 2004 `C30/37` needs fracture energy: an explicitly enabled `G_Ft=0.15`
  constitutive override resolves the representative case.
- EC2 2023 `C30/37` at 56 days reports unresolved `E_initial` plus fracture-energy
  composition. Explicit overrides `E=41000` and `G_Ft=0.15` resolve that case;
  either override alone leaves the other blocker.

These are qualification examples, not recommended project input values. The UI
never guesses fracture energy or substitutes `E_secant` for unresolved `E_initial`.
Common override fields start disabled and blank. Advanced JSON exposes the
existing domain override catalog. All scientific validation remains in the domain.

READY results display all 20 CDPM2 semantic fields with units and provenance.
In **Backend**, supply a positive LCHAR in mm and press **Build backend payload**
to inspect the frozen 24-slot representation. LCHAR is runtime/mesh context and
is displayed separately from intrinsic properties and semantic parameters.

**Raw / Export** provides canonical material JSON, the complete M2 application
result, semantic JSON when READY, and backend JSON when valid runtime context
has been supplied. Physical authority, Grassl calibration, and legacy24
compatibility remain distinct. No modern EC2/fib stress-strain curves or solver
execution are introduced. See [M2 handoff](docs/web_m2_handoff.md).


## Concrete backend successor (WEB-M4)

WEB-M4 keeps physical authority, constitutive calibration and solver-backend
compatibility separate while closing three concrete product gaps.

**fib MC2010 fracture energy when missing.** In the authority/CDPM2 form, enable
**Use fib MC2010 G_F when missing** to apply the verified MC2010 fracture-energy
estimate as a secondary physical authority only when the selected primary profile
has no usable fracture energy. Existing fib materials continue to use their
primary value and no redundant composition is created. The primary EC2 material
serialization remains unresolved/composition-required; only the CDPM2 conversion
receives the separate composition. This option cannot be combined with an
explicit constitutive `G_Ft` override.

**CDPM2 plots.** READY results now show the exact resolved bilinear tensile
softening input law as tensile stress versus crack opening using the semantic
landmarks `(0,f_t)`, `(w_f1,f_t1)`, `(w_f,0)`. The application verifies that the
polyline area equals `G_Ft`. With a backend LCHAR, it also shows a regularized
crack-band strain view `w/LCHAR`. These are visualizations of resolved input
semantics, **not** numerical integration of a uniaxial CDPM2 loading path; no
compression response or damage history is invented.

**Legacy Abaqus CDP.** The Legacy curves workflow now includes an **Abaqus CDP**
tab. It preserves the repository's historical `dilation_angle`, `fbfc`, `Kc`,
compression/tension curves and damage laws, wraps them in machine-readable
provenance, and exports bilinear or power-law tension as
`*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT`. Abaqus-documented defaults
`eccentricity=0.1` and `viscosity=0` are explicit backend settings and may be
overridden. Backend-only compatibility normalization forces the first compression
damage point to zero and caps damage at 0.99; the legacy arrays themselves are
unchanged.

Abaqus **REF LENGTH** is exposed separately as a damage-conversion reference
length (default `1.0 mm` in this repository's N-mm-MPa convention); it is not the
legacy crack-band `l_ch`. Before `.inp` export the service validates the documented
plastic-strain/plastic-displacement conversions and refuses invalid executable
cards. The JSON download carries all values, tables, provenance, normalization
records, validation and documentation references. This is a **legacy compatibility
calibration**: verify it against project calibration, experiments and the Abaqus
version actually used. WEB-M4 does not run Abaqus and does not export dependent
rate/temperature damage tables (dependent hardening/stiffening tables were added in
ABAQUS-Q1, below). See [WEB-M4 research basis](docs/research/abaqus_cdp_backend.md).


## Steel and comparison workspace (WEB-M3)

The same application now routes four workflows: **Legacy curves**,
**Authority-aware material / CDPM2**, **Steel Johnson-Cook**, and **Compare**.
Install and launch with the existing optional extra; no new dependencies:

```bash
uv sync --locked --extra web
uv run --locked --extra web streamlit run cdp_generator/web/app.py
```

Steel selects EC2, ACI or NCh grades from the existing database, or accepts a
custom material with explicit `fu` or `fu/fy`. **Built-in approximate preset —
verify against the applicable standard / project data.** These presets are not
verified normative authorities. Explicit property overrides remain identified
in the result; custom inputs are labeled `user_provided`, not verified.

The existing Johnson-Cook calibrator supplies all eight parameters. Neutral
rate/temperature behavior defaults to `C=m=0`. Disable neutral to enter `C` and
`m`; select explicit `n` or retain the existing automatic heuristic (not an
experimental fit). Choose rates, temperatures, point count, strain limit and
true/engineering output, then press **Calculate steel**. Changing controls does
not calculate automatically; invalid attempts preserve the displayed result.
Inspect Plotly curves, resolved material/metadata, calibration, parameters and
raw data. Download strict JSON, the historical XLSX workbook, or ABAQUS text.
**Experimental template — verify against the ABAQUS documentation/version used.**
The filesystem export APIs, interactive `cdp-steel`, and Matplotlib remain valid.
The XLSX retains its historical common-grid interpolation; canonical JSON and
Plotly arrays are copied directly without interpolation or resampling.

Every result has **Add to Compare** in the sidebar. Enter a unique, editable
case label to save a snapshot, with up to four cases per family. **Compare**
allows selecting cases, renaming/removing cases and clearing the selected family.
Cases last only for the current session. Legacy concrete, authority-aware
concrete and steel remain separate. Curve overlays require equal quantities,
units and stress representations. Engineering total strain is not true total
strain; plastic strain stays true in both outputs, while its stress representation
still differs. No comparison-layer conversion is performed.

Authority comparison shows physical values/units/resolution (`—` for unresolved
values), readiness/blockers, and semantic CDPM2 parameters only when at least
two selected cases are READY. LCHAR and legacy24 slots are excluded. Steel
comparison shows material/JC values, units and approximate/custom status.
Comparison is descriptive: it produces no rankings or material recommendations.

See [M3 handoff](docs/web_m3_handoff.md) for qualification and the remaining
TS-GATE. Authentication/users, persistent projects, shareable URLs, complex
client interactions, large responsive interfaces, background jobs or integration
into a broader web product could justify a FastAPI + React/TypeScript migration.
Until those needs arise, Streamlit remains a valid internal scientific frontend.

## Abaqus dependent tables and solver qualification (ABAQUS-Q1 / WEB-M5)

The Legacy curves **Abaqus CDP** tab offers three export modes:

- **Static reference** — the unchanged WEB-M4 material.
- **Strain-rate dependent** — exact legacy rate families as
  `*CONCRETE COMPRESSION HARDENING` (stress, inelastic strain, rate) and
  `*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT` (stress, crack opening, crack-opening rate).
  The compression rate column is the legacy curve-family control rate (an explicit *legacy
  rate-axis mapping*, not a reconstructed inelastic strain rate); the tension rate is the
  existing legacy mapping `w_dot = strain_rate × l_ch` [mm/s].
- **Temperature dependent** — exact legacy temperature families with temperature columns and a
  temperature-dependent `*ELASTIC` table `E(T), nu, T`. `E(T)` is the legacy modulus used to
  build the exported inelastic strains; `nu` is held constant (legacy constant-nu assumption).

Damage policy for dependent exports: **omit** (default) or **reuse reference damage if
Abaqus-valid**. Abaqus damage has no rate column and the legacy kernel has no `damage(T)`, so
reference damage is one reused function and is validated against every family with that
family's modulus; incompatible combinations are rejected (for the default rates 0/2/30/100 1/s
and for the temperature cases they are). No rate × temperature surfaces are produced.

```python
from cdp_generator.application import (
    AbaqusLegacyDependentRequest, run_abaqus_legacy_dependent_material,
    abaqus_legacy_dependent_material_text,
)
result = run_abaqus_legacy_dependent_material(
    AbaqusLegacyDependentRequest(mode="strain_rate", strain_rates=(0.0, 2.0, 30.0, 100.0))
)
print(abaqus_legacy_dependent_material_text(result))
```

Optional real-solver gate (never part of plain `pytest -q`):

```bash
uv run python scripts/qualify_abaqus.py          # exit 0 PASS, 1 FAIL, 3 NOT_AVAILABLE
uv run pytest -q -m abaqus_external
```

`NOT_AVAILABLE` is never reported as a pass. A PASS shows only that the keywords are accepted
and that single-element compression/tension paths execute with consistent state variables; it
says nothing about experimental, structural or normative validity. See
[qualification/abaqus/README.md](qualification/abaqus/README.md),
[dependent-table research](docs/research/abaqus_dependent_tables.md) and the
[ABAQUS-Q1 handoff](docs/abaqus_q1_handoff.md).
