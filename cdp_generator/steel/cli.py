"""
Command-line interface for steel Johnson-Cook analysis.

This CLI provides an interactive interface for:
- Selecting steel standards and grades
- Customizing material properties
- Generating stress-strain curves at different rates and temperatures
- Exporting results to Excel
- Visualizing results
"""

import sys
from typing import Optional, List

from .johnson_cook import generate_jc_curves_multicase
from .standards import (
    get_steel_spec,
    calibrate_jc_from_spec,
    print_standards_info,
    list_available_standards,
)
from .export import export_steel_to_excel, print_steel_properties, export_abaqus_material_card
from .plotting import plot_steel_results


def parse_float_list(s: str) -> List[float]:
    """Parse comma-separated float list."""
    try:
        return [float(x.strip()) for x in s.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid number in list: {s}") from e


def get_user_input(prompt: str, default: Optional[str] = None, type_func=str):
    """
    Get user input with optional default value.

    Args:
        prompt: Prompt message
        default: Default value (shown in prompt)
        type_func: Function to convert input (str, int, float, etc.)

    Returns:
        User input converted with type_func
    """
    if default is not None:
        prompt_with_default = f"{prompt} [{default}]: "
    else:
        prompt_with_default = f"{prompt}: "

    user_input = input(prompt_with_default).strip()

    if not user_input and default is not None:
        return type_func(default)
    elif not user_input:
        raise ValueError("Input required")
    else:
        return type_func(user_input)


def main():
    """Main CLI entry point for steel analysis."""
    print("\n" + "=" * 70)
    print("STEEL JOHNSON-COOK ANALYSIS")
    print("CDP-Generator - Steel Module")
    print("=" * 70)

    try:
        # ====================================================================
        # STEP 1: Standard selection
        # ====================================================================
        print("\n" + "-" * 70)
        print("STEP 1: Steel Standard Selection")
        print("-" * 70)

        standards = list_available_standards()
        print("\nAvailable standards:")
        for std, grades in standards.items():
            print(f"  {std}: {', '.join(grades)}")
        print("  CUSTOM: Define your own properties")

        print("\nTo see detailed information about standards, type 'info'")

        standard = input("\nSelect standard (EC2/ACI/NCh/CUSTOM) [EC2]: ").strip().upper()
        if not standard:
            standard = "EC2"

        if standard == "INFO":
            print_standards_info()
            standard = input("\nSelect standard (EC2/ACI/NCh/CUSTOM) [EC2]: ").strip().upper()
            if not standard:
                standard = "EC2"

        # ====================================================================
        # STEP 2: Grade and properties
        # ====================================================================
        print("\n" + "-" * 70)
        print("STEP 2: Grade and Properties")
        print("-" * 70)

        overrides = {}

        if standard == "CUSTOM":
            print("\nDefine custom steel properties:")
            grade = get_user_input("Grade name", default="CustomSteel")

            fy = get_user_input("Yield strength fy [MPa]", type_func=float)
            overrides['fy'] = fy

            fu_choice = input("Enter (1) ultimate strength fu or (2) fu/fy ratio? [1]: ").strip()
            if fu_choice == "2":
                fu_fy = get_user_input("fu/fy ratio", type_func=float)
                overrides['fu_fy_ratio'] = fu_fy
            else:
                fu = get_user_input("Ultimate strength fu [MPa]", type_func=float)
                overrides['fu'] = fu

            Agt = get_user_input("Elongation at max force Agt [%]", type_func=float)
            overrides['Agt'] = Agt

            E = get_user_input("Elastic modulus E [MPa]", default="200000", type_func=float)
            overrides['E'] = E

        else:
            # Select grade from available
            available_grades = standards[standard]
            print(f"\nAvailable grades for {standard}: {', '.join(available_grades)}")

            if standard == "EC2":
                grade = get_user_input("Grade", default="B500C")
            elif standard == "ACI":
                grade = get_user_input("Grade", default="A615_60")
            elif standard == "NCh":
                grade = get_user_input("Grade", default="A630-420H")
            else:
                grade = get_user_input("Grade")

            # Ask if user wants to override any values
            override_choice = input("\nOverride standard values? (y/n) [n]: ").strip().lower()
            if override_choice == 'y':
                print("\nLeave blank to use standard value")

                fy_input = input("  Override fy [MPa]: ").strip()
                if fy_input:
                    overrides['fy'] = float(fy_input)

                fu_input = input("  Override fu [MPa]: ").strip()
                if fu_input:
                    overrides['fu'] = float(fu_input)

                Agt_input = input("  Override Agt [%]: ").strip()
                if Agt_input:
                    overrides['Agt'] = float(Agt_input)

                E_input = input("  Override E [MPa]: ").strip()
                if E_input:
                    overrides['E'] = float(E_input)

        # Get steel spec
        spec = get_steel_spec(
            standard=standard,
            grade=grade,
            overrides=overrides if overrides else None
        )

        # ====================================================================
        # STEP 3: Johnson-Cook calibration parameters
        # ====================================================================
        print("\n" + "-" * 70)
        print("STEP 3: Johnson-Cook Calibration")
        print("-" * 70)

        print("\nCalibration options:")
        print("  The JC parameters A, B, n will be calibrated from fy, fu, Agt")
        print("  Rate and temperature parameters C and m can be:")
        print("    - Set to 0 (rate/temp neutral)")
        print("    - Set to custom values")

        rate_temp_choice = input("\nAssume rate/temp neutral (C=0, m=0)? (y/n) [y]: ").strip().lower()
        assume_neutral = rate_temp_choice != 'n'

        C_val = 0.0
        m_val = 0.0
        if not assume_neutral:
            C_val = get_user_input("Strain rate coefficient C", default="0.01", type_func=float)
            m_val = get_user_input("Thermal softening exponent m", default="1.0", type_func=float)

        n_input = input("Hardening exponent n (leave blank for auto) [auto]: ").strip()
        n_val = float(n_input) if n_input else None

        epsdot0 = get_user_input("Reference strain rate ε̇₀ [1/s]", default="1e-3", type_func=float)

        # Calibrate
        print("\nCalibrating Johnson-Cook parameters...")
        params = calibrate_jc_from_spec(
            spec,
            assume_rate_temp_neutral=assume_neutral,
            n_default=n_val,
            C_default=C_val,
            m_default=m_val,
            epsdot0=epsdot0,
            verbose=True
        )

        # ====================================================================
        # STEP 4: Analysis parameters
        # ====================================================================
        print("\n" + "-" * 70)
        print("STEP 4: Analysis Parameters")
        print("-" * 70)

        print("\nStrain rates (comma-separated list):")
        strain_rates_input = get_user_input(
            "Strain rates [1/s]",
            default="1e-4,1e-3,1e-2,1,10,100"
        )
        strain_rates = parse_float_list(strain_rates_input)

        print("\nTemperatures (comma-separated list):")
        temperatures_input = get_user_input(
            "Temperatures [°C]",
            default="20,200,400,600,800"
        )
        temperatures = parse_float_list(temperatures_input)

        eps_max = get_user_input("Maximum strain", default="0.20", type_func=float)
        n_points = get_user_input("Number of points per curve", default="100", type_func=int)

        output_kind_choice = input("Output type: (1) True or (2) Engineering? [1]: ").strip()
        output_kind = "engineering" if output_kind_choice == "2" else "true"

        # ====================================================================
        # STEP 5: Generate curves
        # ====================================================================
        print("\n" + "-" * 70)
        print("STEP 5: Generating Curves")
        print("-" * 70)

        print(f"\nGenerating {len(strain_rates) * len(temperatures)} curves...")

        results = generate_jc_curves_multicase(
            params=params,
            E=spec.E,
            eps_max=eps_max,
            n_points=n_points,
            strain_rates=strain_rates,
            temperatures=temperatures,
            output_kind=output_kind
        )

        print(f"✓ Generated {len(results['curves'])} curves")

        # ====================================================================
        # STEP 6: Export
        # ====================================================================
        print("\n" + "-" * 70)
        print("STEP 6: Export Results")
        print("-" * 70)

        excel_filename = get_user_input(
            "Excel filename",
            default="Steel-JC-Results.xlsx"
        )

        export_steel_to_excel(results, filename=excel_filename, verbose=True)

        # Optional ABAQUS export
        abaqus_choice = input("\nExport ABAQUS material card? (y/n) [n]: ").strip().lower()
        if abaqus_choice == 'y':
            abaqus_filename = get_user_input(
                "ABAQUS filename",
                default="steel_abaqus_material.inp"
            )
            export_abaqus_material_card(
                params,
                E=spec.E,
                nu=spec.nu,
                filename=abaqus_filename
            )

        # ====================================================================
        # STEP 7: Plotting
        # ====================================================================
        print("\n" + "-" * 70)
        print("STEP 7: Plotting")
        print("-" * 70)

        plot_choice = input("\nShow plots? (y/n) [y]: ").strip().lower()
        if plot_choice != 'n':
            print("\nGenerating plots...")
            plot_steel_results(results, show=True)

        # ====================================================================
        # Done
        # ====================================================================
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nResults exported to: {excel_filename}")
        print("\nThank you for using CDP-Generator Steel Module!")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
