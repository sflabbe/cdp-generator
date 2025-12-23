"""
Plotting functions for Johnson-Cook steel stress-strain curves.

This module provides visualization for steel behavior under different
strain rates and temperatures, following the style of the CDP plotting module.
"""

from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import numpy as np


# Color and marker cycles (matching CDP style)
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
]

MARKERS = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '<', '>', '8']


def plot_steel_results(
    results: Dict[str, Any],
    mode: str = "auto",
    show: bool = True,
    save_prefix: Optional[str] = None
) -> None:
    """
    Plot all steel results from Johnson-Cook analysis.

    Args:
        results: Results dictionary from generate_jc_curves_multicase()
        mode: Plotting mode:
            - "auto": Detect from data (strain_rate or temperature)
            - "strain_rate": Plot vs. strain rate
            - "temperature": Plot vs. temperature
        show: Show plots in interactive windows (default: True)
        save_prefix: If provided, save plots to files with this prefix

    Creates plots:
        1. Engineering stress-strain (if available)
        2. True stress-strain (if available)
        3. Stress vs. plastic strain
        4. Yield strength vs. strain rate (if multiple rates)
        5. Yield strength vs. temperature (if multiple temps)
    """
    curves = results['curves']
    strain_rates = results['strain_rates']
    temperatures = results['temperatures']
    params = results['params']

    # Determine mode
    if mode == "auto":
        if len(strain_rates) > 1 and len(temperatures) == 1:
            mode = "strain_rate"
        elif len(temperatures) > 1 and len(strain_rates) == 1:
            mode = "temperature"
        elif len(strain_rates) > 1 and len(temperatures) > 1:
            mode = "both"
        else:
            mode = "single"

    # Check what output types we have
    has_eng = any(c['output_kind'] == 'engineering' for c in curves)
    has_true = any(c['output_kind'] == 'true' for c in curves)

    # Plot 1: Engineering stress-strain
    if has_eng:
        eng_curves = [c for c in curves if c['output_kind'] == 'engineering']
        _plot_stress_strain_curves(
            eng_curves,
            mode=mode,
            title="Engineering Stress-Strain Curves",
            xlabel="Engineering Strain [-]",
            ylabel="Engineering Stress [MPa]",
            save_path=f"{save_prefix}_eng_stress_strain.png" if save_prefix else None
        )

    # Plot 2: True stress-strain
    if has_true:
        true_curves = [c for c in curves if c['output_kind'] == 'true']
        _plot_stress_strain_curves(
            true_curves,
            mode=mode,
            title="True Stress-Strain Curves",
            xlabel="True Strain [-]",
            ylabel="True Stress [MPa]",
            save_path=f"{save_prefix}_true_stress_strain.png" if save_prefix else None
        )

    # Plot 3: Stress vs. plastic strain
    _plot_stress_plastic_curves(
        curves,
        mode=mode,
        title="Stress vs. Plastic Strain",
        xlabel="Plastic Strain [-]",
        ylabel="Stress [MPa]",
        save_path=f"{save_prefix}_plastic_strain.png" if save_prefix else None
    )

    # Plot 4: Yield strength vs. strain rate (if applicable)
    if len(strain_rates) > 1 and params.C != 0:
        _plot_rate_effect(
            strain_rates,
            params,
            temperatures[0],
            save_path=f"{save_prefix}_rate_effect.png" if save_prefix else None
        )

    # Plot 5: Yield strength vs. temperature (if applicable)
    if len(temperatures) > 1 and params.m != 0:
        _plot_temperature_effect(
            temperatures,
            params,
            strain_rates[0],
            save_path=f"{save_prefix}_temp_effect.png" if save_prefix else None
        )

    if show:
        plt.show()


def _plot_stress_strain_curves(
    curves: List[Dict],
    mode: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[str] = None
) -> None:
    """Plot stress-strain curves for multiple cases."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, curve in enumerate(curves):
        strain = curve['strain']
        stress = curve['stress']
        epsdot = curve['epsdot']
        T = curve['T']

        # Create label based on mode
        if mode == "strain_rate":
            label = f"$\\dot{{\\varepsilon}}$ = {epsdot:.2e} 1/s"
        elif mode == "temperature":
            label = f"T = {T:.1f} °C"
        elif mode == "both":
            label = f"$\\dot{{\\varepsilon}}$ = {epsdot:.2e}, T = {T:.1f} °C"
        else:
            label = curve.get('case_id', f"Case {i+1}")

        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        # Plot with markers at reduced frequency for clarity
        marker_every = max(1, len(strain) // 10)
        ax.plot(strain, stress, label=label, color=color, linewidth=2,
                marker=marker, markevery=marker_every, markersize=6)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")


def _plot_stress_plastic_curves(
    curves: List[Dict],
    mode: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[str] = None
) -> None:
    """Plot stress vs. plastic strain curves."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, curve in enumerate(curves):
        eps_p = curve['plastic_strain']
        stress = curve['stress']
        epsdot = curve['epsdot']
        T = curve['T']

        # Create label
        if mode == "strain_rate":
            label = f"$\\dot{{\\varepsilon}}$ = {epsdot:.2e} 1/s"
        elif mode == "temperature":
            label = f"T = {T:.1f} °C"
        elif mode == "both":
            label = f"$\\dot{{\\varepsilon}}$ = {epsdot:.2e}, T = {T:.1f} °C"
        else:
            label = curve.get('case_id', f"Case {i+1}")

        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        # Only plot where eps_p > 0 (plastic region)
        mask = eps_p > 1e-8
        if np.any(mask):
            marker_every = max(1, np.sum(mask) // 10)
            ax.plot(eps_p[mask], stress[mask], label=label, color=color, linewidth=2,
                    marker=marker, markevery=marker_every, markersize=6)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")


def _plot_rate_effect(
    strain_rates: np.ndarray,
    params,
    T: float,
    save_path: Optional[str] = None
) -> None:
    """Plot yield strength vs. strain rate."""
    from .johnson_cook import johnson_cook_flow_stress

    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate yield strength (at eps_p = 0) for each rate
    sigma_y = np.array([
        johnson_cook_flow_stress(0.0, epsdot, T, params)[0]
        for epsdot in strain_rates
    ])

    ax.semilogx(strain_rates, sigma_y, 'o-', linewidth=2, markersize=8,
                color=COLORS[0], label='Yield Strength')

    # Add reference line at A
    ax.axhline(params.A, color='gray', linestyle='--', linewidth=1.5,
               label=f'A = {params.A:.0f} MPa (at $\\dot{{\\varepsilon}}_0$)')

    ax.set_xlabel('Strain Rate [1/s]', fontsize=12)
    ax.set_ylabel('Yield Strength [MPa]', fontsize=12)
    ax.set_title(f'Strain Rate Effect on Yield Strength (T = {T:.1f} °C)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")


def _plot_temperature_effect(
    temperatures: np.ndarray,
    params,
    epsdot: float,
    save_path: Optional[str] = None
) -> None:
    """Plot yield strength vs. temperature."""
    from .johnson_cook import johnson_cook_flow_stress

    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate yield strength (at eps_p = 0) for each temperature
    sigma_y = np.array([
        johnson_cook_flow_stress(0.0, epsdot, T, params)[0]
        for T in temperatures
    ])

    ax.plot(temperatures, sigma_y, 'o-', linewidth=2, markersize=8,
            color=COLORS[1], label='Yield Strength')

    # Add reference line at A
    ax.axhline(params.A, color='gray', linestyle='--', linewidth=1.5,
               label=f'A = {params.A:.0f} MPa (at T_room)')

    ax.set_xlabel('Temperature [°C]', fontsize=12)
    ax.set_ylabel('Yield Strength [MPa]', fontsize=12)
    ax.set_title(f'Temperature Effect on Yield Strength ($\\dot{{\\varepsilon}}$ = {epsdot:.2e} 1/s)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")


def plot_single_curve(
    curve: Dict[str, Any],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot a single stress-strain curve.

    Args:
        curve: Single curve dictionary from generate_jc_curve()
        title: Plot title (auto-generated if None)
        save_path: Save path for figure (optional)
        show: Show plot interactively
    """
    strain = curve['strain']
    stress = curve['stress']
    output_kind = curve['output_kind']
    epsdot = curve['epsdot']
    T = curve['T']

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(strain, stress, linewidth=2.5, color=COLORS[0])

    if output_kind == 'engineering':
        xlabel = 'Engineering Strain [-]'
        ylabel = 'Engineering Stress [MPa]'
    else:
        xlabel = 'True Strain [-]'
        ylabel = 'True Stress [MPa]'

    if title is None:
        title = f"Steel Stress-Strain Curve\n$\\dot{{\\varepsilon}}$ = {epsdot:.2e} 1/s, T = {T:.1f} °C"

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    if show:
        plt.show()


def compare_standards(
    results_list: List[Dict[str, Any]],
    labels: List[str],
    save_prefix: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Compare stress-strain curves from different standards or configurations.

    Args:
        results_list: List of results dictionaries
        labels: List of labels for each result set
        save_prefix: Prefix for saved figures
        show: Show plots interactively
    """
    if len(results_list) != len(labels):
        raise ValueError("results_list and labels must have same length")

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (results, label) in enumerate(zip(results_list, labels)):
        # Plot first curve from each result set
        curve = results['curves'][0]
        strain = curve['strain']
        stress = curve['stress']

        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        marker_every = max(1, len(strain) // 10)

        ax.plot(strain, stress, label=label, color=color, linewidth=2,
                marker=marker, markevery=marker_every, markersize=6)

    ax.set_xlabel('Strain [-]', fontsize=12)
    ax.set_ylabel('Stress [MPa]', fontsize=12)
    ax.set_title('Comparison of Steel Standards', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if save_prefix:
        save_path = f"{save_prefix}_standards_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    if show:
        plt.show()
