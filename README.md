# CDP Generator

A comprehensive Python package for generating material model parameters for ABAQUS finite element simulations:

1. **Concrete Damage Plasticity (CDP)** - For concrete materials
2. **Johnson-Cook (JC) Plasticity** - For steel materials

Both models support strain-rate and temperature-dependent properties.

## Features

### Concrete (CDP)
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Strain Rate Dependent Analysis**: Calculates CDP parameters for multiple strain rates
- **Temperature Dependent Analysis**: Computes temperature-dependent properties based on Eurocode
- **Multiple Models**: Supports both bilinear and power law tension softening models
- **Comprehensive Output**: Generates stress-strain curves, damage parameters, and material properties

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

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/sflabbe/cdp-generator.git
cd cdp-generator

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### From PyPI (when published)

```bash
pip install cdp-generator
```

### For Use in Other Projects

```bash
# Install from GitHub
pip install git+https://github.com/sflabbe/cdp-generator.git

# Or add to requirements.txt
# git+https://github.com/sflabbe/cdp-generator.git
```

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

