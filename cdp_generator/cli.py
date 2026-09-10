"""
Command Line Interface Module

Interactive CLI for generating CDP parameters.
"""

from .core import calculate_stress_strain, calculate_stress_strain_temp
from .export import export_to_excel, print_properties
from .plotting import plot_all_results


def main() -> None:
    """
    Main CLI function for interactive CDP parameter generation.
    """
    print("CDP Generator - Concrete Damage Plasticity Model Input Parameter Generator")
    print("=" * 80)

    # Get user inputs with defaults. Keep raw input strings separate from parsed
    # numeric values so static typing reflects the actual CLI data flow.
    f_cm_input = input("Enter the compressive strength of the concrete (MPa) [Default: 28]: ")
    e_c1_input = input("Enter the strain at maximum compressive strength c_i [Default: 0.0022]: ")
    e_clim_input = input("Enter the strain at ultimate state [Default: 0.0035]: ")
    l_ch_input = input("Enter the characteristic element length of the mesh (mm) [Default: 1]: ")
    e_rate_input = input(
        "Enter the strain rates additional to 0/s, separated by a comma [Default: 2,30,100]: "
    )

    while True:
        temp_input = input("Temperature Dependent Data? (y/n) [Default: n]: ").strip().lower()
        if temp_input in ("y", "n", ""):
            is_strain_rate_mode = temp_input != "y"
            break
        print("Invalid input. Please enter 'y' or 'n'.")

    # Assign default values if no input is provided.
    f_cm = float(f_cm_input.strip()) if f_cm_input.strip() else 28.0
    e_c1 = float(e_c1_input.strip()) if e_c1_input.strip() else 0.0022
    e_clim = float(e_clim_input.strip()) if e_clim_input.strip() else 0.0035
    l_ch = float(l_ch_input.strip()) if l_ch_input.strip() else 1.0
    strain_rates = (
        [0.0, *(float(rate) for rate in e_rate_input.strip().split(","))]
        if e_rate_input.strip()
        else [0.0, 2.0, 30.0, 100.0]
    )
    temperatures = [
        20.0,
        100.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
        1000.0,
        1100.0,
    ]

    # Calculate stress-strain relationships.
    if is_strain_rate_mode:
        results = calculate_stress_strain(f_cm, e_c1, e_clim, l_ch, strain_rates)
        var = strain_rates
        mode = "strain_rate"
    else:
        results = calculate_stress_strain_temp(f_cm, e_c1, e_clim, l_ch)
        var = temperatures
        mode = "temperature"

    # Plot results.
    plot_all_results(results, var, mode)

    # Print properties.
    print_properties(f_cm, results)

    # Export to Excel.
    print("\nExporting results to Excel...")
    excel_file = export_to_excel(results, var, mode)
    print(f"Results exported successfully to {excel_file}")


if __name__ == "__main__":
    main()
