"""
Export Johnson-Cook steel results to Excel format.

This module provides Excel export functionality for steel stress-strain curves
and material properties, following the style of the CDP concrete export.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from .johnson_cook import JohnsonCookParams
from .standards import SteelSpec


def export_steel_to_excel(
    results: Dict[str, Any],
    filename: str = "Steel-JC-Results.xlsx",
    verbose: bool = True
) -> None:
    """
    Export steel Johnson-Cook results to Excel file.

    Args:
        results: Results dictionary from generate_jc_curves_multicase() with:
            - 'curves': List of curve dictionaries
            - 'strain_rates': Array of strain rates
            - 'temperatures': Array of temperatures
            - 'params': JohnsonCookParams
            - 'E': Elastic modulus
        filename: Output Excel filename (default: "Steel-JC-Results.xlsx")
        verbose: Print export progress (default: True)

    Sheets created:
        1. "Steel Properties" - Material properties and JC parameters
        2. "JC Parameters" - Detailed Johnson-Cook parameters
        3. "Stress-Strain (eng)" - Engineering stress-strain curves
        4. "Stress-Strain (true)" - True stress-strain curves
        5. "Plastic Strain" - Stress vs. plastic strain curves
        6. "Summary" - Overview of all cases

    Notes:
        - All curves are labeled with strain rate and temperature
        - Supports multiple strain rates and temperatures
        - Compatible with Excel 2007+ (.xlsx format)
    """
    if verbose:
        print(f"\nExporting steel results to {filename}...")

    curves = results['curves']
    params = results['params']
    E = results['E']
    strain_rates = results['strain_rates']
    temperatures = results['temperatures']

    # Prepare Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:

        # ====================================================================
        # Sheet 1: Steel Properties
        # ====================================================================
        props_data = {
            'Property': [
                'Elastic Modulus',
                'Reference Strain Rate',
                'Room Temperature',
                'Melting Temperature',
                'Number of Cases',
                'Number of Strain Rates',
                'Number of Temperatures',
            ],
            'Value': [
                f"{E:.0f} MPa",
                f"{params.epsdot0:.2e} 1/s",
                f"{params.T_room:.1f} °C",
                f"{params.T_melt:.1f} °C",
                len(curves),
                len(strain_rates),
                len(temperatures),
            ],
            'Description': [
                'Young\'s modulus',
                'Reference strain rate for JC model',
                'Reference temperature',
                'Material melting temperature',
                'Total number of cases analyzed',
                'Strain rates analyzed',
                'Temperatures analyzed',
            ]
        }
        df_props = pd.DataFrame(props_data)
        df_props.to_excel(writer, sheet_name='Steel Properties', index=False)

        if verbose:
            print("  ✓ Sheet 'Steel Properties' created")

        # ====================================================================
        # Sheet 2: JC Parameters
        # ====================================================================
        jc_data = {
            'Parameter': ['A', 'B', 'n', 'C', 'm', 'ε̇₀', 'T_room', 'T_melt'],
            'Value': [
                params.A,
                params.B,
                params.n,
                params.C,
                params.m,
                params.epsdot0,
                params.T_room,
                params.T_melt,
            ],
            'Unit': ['MPa', 'MPa', '-', '-', '-', '1/s', '°C', '°C'],
            'Description': [
                'Yield stress',
                'Strain hardening coefficient',
                'Strain hardening exponent',
                'Strain rate sensitivity coefficient',
                'Thermal softening exponent',
                'Reference strain rate',
                'Room temperature',
                'Melting temperature',
            ]
        }
        df_jc = pd.DataFrame(jc_data)
        df_jc.to_excel(writer, sheet_name='JC Parameters', index=False)

        if verbose:
            print("  ✓ Sheet 'JC Parameters' created")

        # ====================================================================
        # Sheet 3: Summary of Cases
        # ====================================================================
        summary_data = []
        for i, curve in enumerate(curves):
            summary_data.append({
                'Case ID': i + 1,
                'Label': curve.get('case_id', f"Case {i+1}"),
                'Strain Rate [1/s]': curve['epsdot'],
                'Temperature [°C]': curve['T'],
                'Output Type': curve['output_kind'],
                'Max Strain [-]': curve['strain'][-1],
                'Max Stress [MPa]': curve['stress'][-1],
            })
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

        if verbose:
            print("  ✓ Sheet 'Summary' created")

        # ====================================================================
        # Sheet 4-6: Stress-Strain Curves
        # ====================================================================
        # We'll create separate sheets for:
        # - Engineering stress-strain (if any curves are in engineering)
        # - True stress-strain (if any curves are in true)
        # - Stress vs plastic strain

        # Check if we have engineering or true curves
        has_eng = any(c['output_kind'] == 'engineering' for c in curves)
        has_true = any(c['output_kind'] == 'true' for c in curves)

        # Engineering stress-strain
        if has_eng:
            df_eng = _build_curve_dataframe(
                curves=[c for c in curves if c['output_kind'] == 'engineering'],
                x_key='strain',
                y_key='stress',
                x_label='Engineering Strain',
                y_label='Engineering Stress [MPa]'
            )
            df_eng.to_excel(writer, sheet_name='Stress-Strain (eng)', index=False)
            if verbose:
                print("  ✓ Sheet 'Stress-Strain (eng)' created")

        # True stress-strain
        if has_true:
            df_true = _build_curve_dataframe(
                curves=[c for c in curves if c['output_kind'] == 'true'],
                x_key='strain',
                y_key='stress',
                x_label='True Strain',
                y_label='True Stress [MPa]'
            )
            df_true.to_excel(writer, sheet_name='Stress-Strain (true)', index=False)
            if verbose:
                print("  ✓ Sheet 'Stress-Strain (true)' created")

        # Stress vs Plastic Strain (always in true plastic strain)
        df_plastic = _build_curve_dataframe(
            curves=curves,
            x_key='plastic_strain',
            y_key='stress',
            x_label='Plastic Strain',
            y_label='Stress [MPa]'
        )
        df_plastic.to_excel(writer, sheet_name='Stress-PlasticStrain', index=False)
        if verbose:
            print("  ✓ Sheet 'Stress-PlasticStrain' created")

        # ====================================================================
        # Sheet: Strain Rates and Temperatures Lists
        # ====================================================================
        params_list_data = {
            'Strain Rates [1/s]': list(strain_rates) + [''] * (len(temperatures) - len(strain_rates))
                if len(temperatures) > len(strain_rates)
                else list(strain_rates),
            'Temperatures [°C]': list(temperatures) + [''] * (len(strain_rates) - len(temperatures))
                if len(strain_rates) > len(temperatures)
                else list(temperatures),
        }
        df_params_list = pd.DataFrame(params_list_data)
        df_params_list.to_excel(writer, sheet_name='Cases Parameters', index=False)
        if verbose:
            print("  ✓ Sheet 'Cases Parameters' created")

    if verbose:
        print(f"\n✅ Excel file saved: {filename}")
        print(f"   Total sheets: {3 + int(has_eng) + int(has_true) + 2}")


def _build_curve_dataframe(
    curves: list,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str
) -> pd.DataFrame:
    """
    Build a pandas DataFrame for multiple curves.

    Args:
        curves: List of curve dictionaries
        x_key: Key for x-axis data in curve dict (e.g., 'strain')
        y_key: Key for y-axis data in curve dict (e.g., 'stress')
        x_label: Column label for x-axis
        y_label: Base column label for y-axis (will be appended with case info)

    Returns:
        DataFrame with x column and multiple y columns (one per curve)
    """
    if not curves:
        return pd.DataFrame()

    # Find maximum number of points across all curves
    max_points = max(len(c[x_key]) for c in curves)

    # Create interpolated common x-axis
    # Use the range of the first curve as reference
    x_min = curves[0][x_key][0]
    x_max = max(c[x_key][-1] for c in curves)
    x_common = np.linspace(x_min, x_max, max_points)

    # Build DataFrame
    data = {x_label: x_common}

    for i, curve in enumerate(curves):
        # Interpolate curve onto common x-axis
        y_interp = np.interp(x_common, curve[x_key], curve[y_key])

        # Create column label with case info
        case_label = curve.get('case_id', f"Case {i+1}")
        epsdot = curve['epsdot']
        T = curve['T']
        col_name = f"{y_label} (ε̇={epsdot:.2e}, T={T:.1f}°C)"

        data[col_name] = y_interp

    return pd.DataFrame(data)


def print_steel_properties(
    spec: Optional[SteelSpec],
    params: JohnsonCookParams,
    E: float
) -> None:
    """
    Print steel properties and Johnson-Cook parameters to console.

    Args:
        spec: SteelSpec (optional, can be None for custom params)
        params: Johnson-Cook parameters
        E: Elastic modulus [MPa]
    """
    print("\n" + "=" * 70)
    print("STEEL MATERIAL PROPERTIES")
    print("=" * 70)

    if spec is not None:
        print(f"\nStandard: {spec.standard}")
        print(f"Grade: {spec.grade}")
        if spec.ductility_class:
            print(f"Ductility Class: {spec.ductility_class}")
        if spec.metadata and 'description' in spec.metadata:
            print(f"Description: {spec.metadata['description']}")

        print(f"\nCharacteristic Properties:")
        print(f"  fy = {spec.fy:.2f} MPa")
        print(f"  fu = {spec.fu:.2f} MPa")
        print(f"  fu/fy = {spec.fu_fy_ratio:.3f}")
        print(f"  Agt = {spec.Agt:.2f} %")

    print(f"\nElastic Properties:")
    print(f"  E = {E:.0f} MPa")
    if spec is not None and spec.nu is not None:
        print(f"  ν = {spec.nu:.3f}")

    print(f"\nJohnson-Cook Parameters:")
    print(f"  A = {params.A:.2f} MPa  (yield stress)")
    print(f"  B = {params.B:.2f} MPa  (hardening coefficient)")
    print(f"  n = {params.n:.4f}  (hardening exponent)")
    print(f"  C = {params.C:.4f}  (rate sensitivity)")
    print(f"  m = {params.m:.4f}  (thermal softening)")
    print(f"  ε̇₀ = {params.epsdot0:.2e} 1/s  (reference rate)")
    print(f"  T_room = {params.T_room:.1f} °C")
    print(f"  T_melt = {params.T_melt:.1f} °C")

    print("\n" + "=" * 70)


def export_abaqus_material_card(
    params: JohnsonCookParams,
    E: float,
    nu: float = 0.30,
    density: Optional[float] = None,
    filename: str = "steel_abaqus_material.inp"
) -> None:
    """
    Export ABAQUS material card for Johnson-Cook plasticity.

    EXPERIMENTAL FEATURE: This generates a basic ABAQUS input format.
    Always verify the output against ABAQUS documentation for your version.

    Args:
        params: Johnson-Cook parameters
        E: Elastic modulus [MPa]
        nu: Poisson's ratio (default: 0.30)
        density: Material density [kg/mm³] (optional)
        filename: Output filename (default: "steel_abaqus_material.inp")

    Notes:
        - Output units: MPa, mm, tonne, s (ABAQUS consistent units)
        - Temperature should be in absolute (Kelvin) for ABAQUS
        - This is a template - adjust for your specific ABAQUS version
    """
    # Convert temperatures to Kelvin for ABAQUS
    T_room_K = params.T_room + 273.15
    T_melt_K = params.T_melt + 273.15

    lines = [
        "*Material, name=Steel_JC",
        "*Elastic",
        f"{E:.1f}, {nu:.3f}",
        "*Plastic, hardening=JOHNSON COOK",
        f"** A,         B,        n,      C,      m,      T_melt,  T_room,   eps_dot_0",
        f"{params.A:.2f}, {params.B:.2f}, {params.n:.4f}, "
        f"{params.C:.4f}, {params.m:.4f}, {T_melt_K:.2f}, {T_room_K:.2f}, {params.epsdot0:.2e}",
    ]

    if density is not None:
        lines.insert(1, "*Density")
        lines.insert(2, f"{density:.2e}")

    content = "\n".join(lines) + "\n"

    with open(filename, 'w') as f:
        f.write(content)

    print(f"\n✅ ABAQUS material card exported: {filename}")
    print("⚠️  IMPORTANT: This is a basic template. Verify against ABAQUS documentation.")
    print("   Temperature units: Kelvin (absolute)")
    print("   Stress units: MPa")
    print("   Strain rate units: 1/s")
