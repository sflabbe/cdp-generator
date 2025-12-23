"""
Steel Johnson-Cook module for cdp-generator.

This subpackage provides Johnson-Cook constitutive modeling for steel materials,
including:
- Standard steel specifications (EC2, ACI, NCh)
- Johnson-Cook parameter calibration
- Stress-strain curve generation
- Excel export
- Visualization

Example usage:
    >>> from cdp_generator.steel import get_steel_spec, calibrate_jc_from_spec
    >>> from cdp_generator.steel import generate_jc_curves_multicase
    >>> from cdp_generator.steel import export_steel_to_excel, plot_steel_results
    >>>
    >>> # Get standard specification
    >>> spec = get_steel_spec("EC2", "B500C")
    >>>
    >>> # Calibrate Johnson-Cook parameters
    >>> params = calibrate_jc_from_spec(spec, verbose=True)
    >>>
    >>> # Generate curves
    >>> results = generate_jc_curves_multicase(
    ...     params, E=200000,
    ...     strain_rates=[1e-4, 1e-2, 1, 100],
    ...     temperatures=[20, 200, 400]
    ... )
    >>>
    >>> # Export and plot
    >>> export_steel_to_excel(results, "my_steel.xlsx")
    >>> plot_steel_results(results)
"""

# Core Johnson-Cook model
from .johnson_cook import (
    JohnsonCookParams,
    johnson_cook_flow_stress,
    eng_to_true,
    true_to_eng,
    true_plastic_strain,
    generate_jc_curve,
    generate_jc_curves_multicase,
)

# Standards and calibration
from .standards import (
    SteelSpec,
    get_steel_spec,
    calibrate_jc_from_spec,
    list_available_standards,
    print_standards_info,
)

# Export functionality
from .export import (
    export_steel_to_excel,
    print_steel_properties,
    export_abaqus_material_card,
)

# Plotting
from .plotting import (
    plot_steel_results,
    plot_single_curve,
    compare_standards,
)

__all__ = [
    # Data classes
    'JohnsonCookParams',
    'SteelSpec',
    # Core functions
    'johnson_cook_flow_stress',
    'eng_to_true',
    'true_to_eng',
    'true_plastic_strain',
    'generate_jc_curve',
    'generate_jc_curves_multicase',
    # Standards
    'get_steel_spec',
    'calibrate_jc_from_spec',
    'list_available_standards',
    'print_standards_info',
    # Export
    'export_steel_to_excel',
    'print_steel_properties',
    'export_abaqus_material_card',
    # Plotting
    'plot_steel_results',
    'plot_single_curve',
    'compare_standards',
]
