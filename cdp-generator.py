"""
CDP Generator - Concrete Damage Plasticity Model Input Parameter Generator

This script generates input parameters for the Concrete Damage Plasticity (CDP)
model in ABAQUS, with support for strain-rate and temperature-dependent properties.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle


# ============================================================================
# Helper Functions - Material Properties
# ============================================================================

def calculate_concrete_strength_properties(f_cm):
    """
    Calculate basic concrete strength properties based on mean compressive strength.

    Args:
        f_cm: Mean compressive strength [MPa]

    Returns:
        dict: Contains f_ck (characteristic strength) and f_ctm (tensile strength)
    """
    f_ck = f_cm - 8

    # Tensile strength according to FIB2010
    if f_ck > 50:
        f_ctm = 2.12 * np.log(1 + 0.1 * f_cm)
    else:
        f_ctm = 0.3 * (f_ck) ** (2/3)

    return {
        'f_ck': f_ck,
        'f_ctm': f_ctm
    }


def calculate_elastic_modulus(f_cm, alpha_E=1.0):
    """
    Calculate elastic modulus based on concrete strength.
    Assumes Quartzite aggregates by default.

    Args:
        f_cm: Mean compressive strength [MPa]
        alpha_E: Aggregate type factor (default 1.0 for Quartzite)

    Returns:
        dict: Contains E_ci (tangent modulus) and E_c (secant modulus)
    """
    E_ci = 21500 * alpha_E * (f_cm / 10) ** (1/3)
    alpha = min(0.8 + 0.2 * f_cm / 88, 1.0)
    E_c = alpha * E_ci

    return {
        'E_ci': E_ci,
        'E_c': E_c
    }


def calculate_poisson_ratios(f_cm, E_c, e_c1):
    """
    Calculate Poisson's ratios at different states.

    Args:
        f_cm: Mean compressive strength [MPa]
        E_c: Elastic modulus [MPa]
        e_c1: Strain at peak compressive strength [-]

    Returns:
        dict: Contains v_c0 (at peak) and v_ce (elastic)
    """
    v_c0 = 0.5  # Poisson's ratio at peak engineering stress
    v_ce = 8e-6 * f_cm**2 + 0.0002 * f_cm + 0.138  # Elastic Poisson's ratio

    return {
        'v_c0': v_c0,
        'v_ce': v_ce
    }


def calculate_cdp_parameters(f_cm, E_c, e_c1, v_c0, v_ce):
    """
    Calculate CDP (Concrete Damage Plasticity) model parameters.

    Args:
        f_cm: Mean compressive strength [MPa]
        E_c: Elastic modulus [MPa]
        e_c1: Strain at peak compressive strength [-]
        v_c0: Poisson's ratio at peak
        v_ce: Elastic Poisson's ratio

    Returns:
        dict: Contains dilation angle, fbfc, and Kc
    """
    # Dilation angle [degrees]
    dilation_angle = np.arctan(
        6 * (v_c0 - v_ce) / (3 * E_c * e_c1 / f_cm + 2 * (v_c0 - v_ce) - 3)
    ) * 180 / np.pi

    # Ratio of biaxial to uniaxial compressive strength
    fbfc = 1.57 * f_cm ** (-0.09)

    # Ratio of second stress invariant on tensile meridian to that on compressive meridian
    K_c = 0.71 * f_cm ** (-0.025)

    return {
        'dilation_angle': dilation_angle,
        'fbfc': fbfc,
        'K_c': K_c
    }


def calculate_fracture_energy(f_cm):
    """
    Calculate fracture energy for concrete.

    Args:
        f_cm: Mean compressive strength [MPa]

    Returns:
        float: Fracture energy [N/mm]
    """
    return 73 * f_cm ** 0.18 / 1000


def calculate_characteristic_length(E_c, G_f, f_ctm):
    """
    Calculate characteristic element length for mesh.

    Args:
        E_c: Elastic modulus [MPa]
        G_f: Fracture energy [N/mm]
        f_ctm: Tensile strength [MPa]

    Returns:
        float: Characteristic length [m]
    """
    return 0.4 * E_c * 1e6 * G_f * 1000 / (f_ctm * 1e6) ** 2


# ============================================================================
# Strain Rate Effects
# ============================================================================

def apply_strain_rate_effects(base_props, strain_rate):
    """
    Apply strain rate effects to material properties.

    Args:
        base_props: Dictionary of base material properties
        strain_rate: Strain rate [1/s]

    Returns:
        dict: Modified properties accounting for strain rate
    """
    f_cm = base_props['f_cm']
    f_ctm = base_props['f_ctm']
    e_c1 = base_props['e_c1']
    E_ci = base_props['E_ci']

    # No strain rate effects for static case
    if strain_rate == 0:
        return {
            'f_cm_dyn': f_cm,
            'f_ctm_dyn': f_ctm,
            'e_c1_dyn': e_c1,
            'E_ci_dyn': E_ci
        }

    # Compressive strength DIF (Dynamic Increase Factor)
    if strain_rate > 30:
        f_cm_dyn = 0.012 * f_cm * (strain_rate / 0.00003) ** (1/3)
    else:
        f_cm_dyn = f_cm * (strain_rate / 0.00003) ** 0.014

    # Tensile strength DIF
    if strain_rate > 10:
        f_ctm_dyn = 0.0062 * f_ctm * (strain_rate / 1e-6) ** (1/3)
    else:
        f_ctm_dyn = f_ctm * (strain_rate / 1e-6) ** 0.018

    # Strain at peak stress
    e_c1_dyn = e_c1 * (strain_rate / 0.00003) ** 0.02

    # Elastic modulus
    E_ci_dyn = E_ci * (strain_rate / 0.00003) ** 0.025

    return {
        'f_cm_dyn': f_cm_dyn,
        'f_ctm_dyn': f_ctm_dyn,
        'e_c1_dyn': e_c1_dyn,
        'E_ci_dyn': E_ci_dyn
    }


def apply_fracture_energy_rate_effects(G_f, strain_rate, l_ch):
    """
    Apply strain rate effects to fracture energy (Li et al.).

    Args:
        G_f: Base fracture energy [N/mm]
        strain_rate: Strain rate [1/s]
        l_ch: Characteristic element length [mm]

    Returns:
        float: Dynamic fracture energy [N/mm]
    """
    if strain_rate == 0:
        return G_f

    w_rate = strain_rate * l_ch

    if w_rate > 200:
        b_g = (200 / 0.01) ** (0.08 - 0.62)
        G_f_dyn = G_f * b_g * (w_rate / 0.01) ** 0.62
    else:
        G_f_dyn = G_f * (w_rate / 0.01) ** 0.08

    return G_f_dyn


# ============================================================================
# Temperature Effects
# ============================================================================

def get_eurocode_temperature_table():
    """
    Return Eurocode temperature-dependent reduction factors.

    Returns:
        ndarray: Temperature table [T, f_ratio, eps_c, eps_cu]
    """
    return np.array([
        [20,   1.00, 0.0025, 0.0200],
        [100,  1.00, 0.0040, 0.0225],
        [200,  0.95, 0.0055, 0.0250],
        [300,  0.85, 0.0070, 0.0275],
        [400,  0.75, 0.0100, 0.0300],
        [500,  0.60, 0.0150, 0.0325],
        [600,  0.45, 0.0250, 0.0350],
        [700,  0.30, 0.0250, 0.0375],
        [800,  0.15, 0.0250, 0.0400],
        [900,  0.08, 0.0250, 0.0425],
        [1000, 0.04, 0.0250, 0.0450],
        [1100, 0.01, 0.0250, 0.0475],
    ])


def apply_temperature_effects(base_props, temperature, temp_table):
    """
    Apply temperature effects to material properties based on Eurocode.

    Args:
        base_props: Dictionary of base material properties
        temperature: Temperature [°C]
        temp_table: Eurocode temperature table

    Returns:
        dict: Modified properties accounting for temperature
    """
    # Find index for this temperature
    temp_idx = np.where(temp_table[:, 0] == temperature)[0][0]

    f_ratio = temp_table[temp_idx, 1]
    eps_c = temp_table[temp_idx, 2]
    eps_cu = temp_table[temp_idx, 3]

    f_ck = base_props['f_ck']
    f_cm = base_props['f_cm']
    f_ctm = base_props['f_ctm']
    E_c1 = base_props['E_c1']

    # Compressive strength at temperature
    f_ck_temp = f_ratio * f_ck
    f_cm_temp = f_ratio * f_cm

    # Tensile strength at temperature (Eurocode approach)
    if temperature <= 100:
        f_ctm_temp_EC = f_ctm
    else:
        f_ctm_temp_EC = max(0, f_ctm * (1 - (temperature - 100) / 500))

    # Tensile strength (FIB approach)
    if f_ck > 50:
        f_ctm_temp = 2.12 * np.log(1 + 0.1 * f_cm_temp)
    else:
        f_ctm_temp = 0.3 * (f_ck_temp) ** (2/3)

    # Strain at peak stress
    e_c1_temp = eps_c

    # Elastic modulus at temperature
    E_c1_temp = (f_ratio / eps_c) / (f_ratio / temp_table[1, 2]) * E_c1

    # Tangent modulus
    k_temp = -0.0193 * f_ck_temp + 2.6408
    E_ci_temp = E_c1_temp * k_temp

    return {
        'f_cm_temp': f_cm_temp,
        'f_ck_temp': f_ck_temp,
        'f_ctm_temp': f_ctm_temp,
        'f_ctm_temp_EC': f_ctm_temp_EC,
        'e_c1_temp': e_c1_temp,
        'eps_cu': eps_cu,
        'E_c1_temp': E_c1_temp,
        'E_ci_temp': E_ci_temp
    }


# ============================================================================
# Compression Behavior
# ============================================================================

def calculate_compression_behavior(f_cm, e_c1, E_ci, E_c1, n_points, e_max):
    """
    Calculate compressive stress-strain behavior using CEB-90 model.

    Args:
        f_cm: Mean compressive strength [MPa]
        e_c1: Strain at peak stress [-]
        E_ci: Tangent modulus [MPa]
        E_c1: Secant modulus [MPa]
        n_points: Number of points in curve
        e_max: Maximum strain to calculate [-]

    Returns:
        dict: Contains strain and stress arrays
    """
    # Strain array
    strain = np.linspace(0, e_max, n_points)

    # CEB-90 parameters
    eta_E = E_ci / E_c1
    e_clim = e_c1 * (0.5 * (0.5 * eta_E + 1) +
                     (0.25 * ((0.5 * eta_E + 1) ** 2) - 0.5) ** 0.5)
    eta = strain / e_c1
    eta_lim = e_clim / e_c1

    xi = 4 * (eta_lim**2 * (eta_E - 2) + 2 * eta_lim - eta_E) / \
         ((eta_lim * (eta_E - 2) + 1) ** 2)

    # Calculate stress
    stress = np.zeros_like(strain)
    for i in range(len(strain)):
        if strain[i] <= e_clim:
            # Ascending branch
            stress[i] = (eta_E * strain[i] / e_c1 - (strain[i] / e_c1) ** 2) / \
                       (1 + (eta_E - 2) * (strain[i] / e_c1)) * f_cm
        else:
            # Descending branch
            stress[i] = f_cm / ((xi / eta_lim - 2 / (eta_lim**2)) *
                               ((strain[i] / e_c1) ** 2) +
                               (4 / eta_lim - xi) * strain[i] / e_c1)

    return {
        'strain': strain,
        'stress': stress,
        'e_clim': e_clim
    }


def calculate_inelastic_compression(strain, stress, f_cm, E_c1):
    """
    Calculate inelastic strain and stress for compression.

    Args:
        strain: Total strain array [-]
        stress: Total stress array [MPa]
        f_cm: Mean compressive strength [MPa]
        E_c1: Secant modulus [MPa]

    Returns:
        dict: Contains inelastic strain and stress arrays
    """
    f_cel = f_cm * 0.4  # Elastic limit

    inelastic_strain = []
    inelastic_stress = []
    first_point = False

    for i in range(len(strain)):
        if stress[i] > f_cel:
            if strain[i] - stress[i] / E_c1 > 0:
                if not first_point:
                    first_point = True
                    inelastic_strain.append(0)
                else:
                    inelastic_strain.append(strain[i] - stress[i] / E_c1)
                inelastic_stress.append(stress[i])
        elif strain[i] > strain[np.argmax(stress)]:
            inelastic_strain.append(strain[i] - stress[i] / E_c1)
            inelastic_stress.append(stress[i])

    return {
        'inelastic_strain': np.array(inelastic_strain),
        'inelastic_stress': np.array(inelastic_stress)
    }


def calculate_compression_damage(inelastic_stress, inelastic_strain, f_cm):
    """
    Calculate compression damage parameter.

    Args:
        inelastic_stress: Inelastic stress array [MPa]
        inelastic_strain: Inelastic strain array [-]
        f_cm: Mean compressive strength [MPa]

    Returns:
        ndarray: Damage array [-]
    """
    damage = np.zeros(len(inelastic_strain))

    for i in range(len(inelastic_strain)):
        if inelastic_stress[i] < f_cm:
            damage[i] = 1 - inelastic_stress[i] / f_cm

    # Ensure damage is monotonically increasing
    damage[np.gradient(damage) < 0] = 0

    return damage


# ============================================================================
# Tension Behavior
# ============================================================================

def calculate_tension_bilinear(f_ctm, G_f, n_points):
    """
    Calculate tension softening using bilinear model (FIB2010).

    Args:
        f_ctm: Tensile strength [MPa]
        G_f: Fracture energy [N/mm]
        n_points: Number of points in curve

    Returns:
        dict: Contains crack opening and stress arrays
    """
    # Characteristic crack openings
    w_1 = G_f / f_ctm
    w_c = 5 * G_f / f_ctm

    # Crack opening array
    crack_opening = np.linspace(0, w_c, n_points)

    # Stress array
    stress = np.zeros(n_points)
    for i in range(n_points):
        if crack_opening[i] < w_1:
            stress[i] = f_ctm * (1 - 0.8 * crack_opening[i] / w_1)
        else:
            stress[i] = f_ctm * (0.25 - 0.05 * crack_opening[i] / w_1)

    return {
        'crack_opening': crack_opening,
        'stress': stress,
        'w_1': w_1,
        'w_c': w_c
    }


def calculate_tension_power_law(f_ctm, G_f, w_c, crack_opening):
    """
    Calculate tension softening using generalized power law.

    Args:
        f_ctm: Tensile strength [MPa]
        G_f: Fracture energy [N/mm]
        w_c: Characteristic crack opening [mm]
        crack_opening: Crack opening array [mm]

    Returns:
        ndarray: Stress array [MPa]
    """
    n_exp = G_f / (f_ctm * w_c - G_f)
    stress = f_ctm * (1 - (crack_opening / w_c) ** n_exp)

    return stress


def calculate_tension_damage(stress, f_ctm):
    """
    Calculate tension damage parameter.

    Args:
        stress: Stress array [MPa]
        f_ctm: Tensile strength [MPa]

    Returns:
        ndarray: Damage array [-]
    """
    return 1 - stress / f_ctm


# ============================================================================
# Main Calculation Functions
# ============================================================================

def calculate_stress_strain(f_cm, e_c1, e_clim, l_ch, strain_rates):
    """
    Calculate stress-strain relationships for different strain rates.

    Args:
        f_cm: Mean compressive strength [MPa]
        e_c1: Strain at peak compressive strength [-]
        e_clim: Ultimate strain [-]
        l_ch: Characteristic element length [mm]
        strain_rates: List of strain rates [1/s]

    Returns:
        dict: Complete results including properties and stress-strain data
    """
    n_points = 20

    # Base material properties
    strength_props = calculate_concrete_strength_properties(f_cm)
    f_ck = strength_props['f_ck']
    f_ctm = strength_props['f_ctm']

    elastic_props = calculate_elastic_modulus(f_cm)
    E_ci = elastic_props['E_ci']
    E_c = elastic_props['E_c']
    E_c1 = f_cm / e_c1

    poisson_props = calculate_poisson_ratios(f_cm, E_c, e_c1)
    v_c0 = poisson_props['v_c0']
    v_ce = poisson_props['v_ce']

    cdp_params = calculate_cdp_parameters(f_cm, E_c, e_c1, v_c0, v_ce)

    G_f = calculate_fracture_energy(f_cm)
    G_c = E_c / (2 * (1 + v_ce))
    l_0 = calculate_characteristic_length(E_c, G_f, f_ctm)

    # Storage for rate-dependent results
    compression_stress = []
    compression_inelastic_strain = []
    compression_inelastic_stress = []
    tension_crack_opening = []
    tension_stress_bilinear = []
    tension_stress_power = []
    tension_cracking_strain = []

    # Base properties for rate calculations
    base_props = {
        'f_cm': f_cm,
        'f_ctm': f_ctm,
        'e_c1': e_c1,
        'E_ci': E_ci
    }

    # Calculate for each strain rate
    for rate in strain_rates:
        # Apply strain rate effects
        rate_props = apply_strain_rate_effects(base_props, rate)
        f_cm_dyn = rate_props['f_cm_dyn']
        f_ctm_dyn = rate_props['f_ctm_dyn']
        e_c1_dyn = rate_props['e_c1_dyn']
        E_ci_dyn = rate_props['E_ci_dyn']
        E_c1_dyn = f_cm_dyn / e_c1_dyn

        # Compression behavior
        comp_result = calculate_compression_behavior(
            f_cm_dyn, e_c1_dyn, E_ci_dyn, E_c1_dyn,
            n_points, e_clim * 3
        )
        compression_stress.append(comp_result['stress'])

        # Inelastic compression
        inel_result = calculate_inelastic_compression(
            comp_result['strain'], comp_result['stress'],
            f_cm_dyn, E_c1_dyn
        )
        compression_inelastic_strain.append(inel_result['inelastic_strain'])
        compression_inelastic_stress.append(inel_result['inelastic_stress'])

        # Tension behavior
        G_f_dyn = apply_fracture_energy_rate_effects(G_f, rate, l_ch)

        # Bilinear tension model
        tension_result = calculate_tension_bilinear(f_ctm_dyn, G_f_dyn, n_points)
        tension_crack_opening.append(tension_result['crack_opening'])
        tension_stress_bilinear.append(tension_result['stress'])

        # Power law tension model
        stress_power = calculate_tension_power_law(
            f_ctm_dyn, G_f_dyn,
            tension_result['w_c'],
            tension_result['crack_opening']
        )
        tension_stress_power.append(stress_power)

        # Cracking strain
        cracking_strain = tension_result['crack_opening'] / l_ch
        tension_cracking_strain.append(cracking_strain)

    # Interpolate all curves to match first strain rate grid
    for i in range(1, len(strain_rates)):
        compression_inelastic_stress[i] = np.interp(
            compression_inelastic_strain[0],
            compression_inelastic_strain[i],
            compression_inelastic_stress[i]
        )
        compression_inelastic_strain[i] = compression_inelastic_strain[0]

        tension_stress_bilinear[i] = np.interp(
            tension_crack_opening[0],
            tension_crack_opening[i],
            tension_stress_bilinear[i]
        )
        tension_stress_power[i] = np.interp(
            tension_crack_opening[0],
            tension_crack_opening[i],
            tension_stress_power[i]
        )
        tension_cracking_strain[i] = tension_cracking_strain[0]
        tension_crack_opening[i] = tension_crack_opening[0]

    # Calculate damage (based on reference rate)
    damage_c = calculate_compression_damage(
        compression_inelastic_stress[0],
        compression_inelastic_strain[0],
        f_cm
    )

    damage_t = calculate_tension_damage(tension_stress_bilinear[0], f_ctm)
    damage_t_power = calculate_tension_damage(tension_stress_power[0], f_ctm)

    return {
        'properties': {
            'elasticity': E_c,
            'shear': G_c,
            'fracture energy': G_f,
            'tensile strength': f_ctm,
            'dilation angle': cdp_params['dilation_angle'],
            'poisson': v_ce,
            'Kc': cdp_params['K_c'],
            'fbfc': cdp_params['fbfc'],
            'l0': l_0
        },
        'compression': {
            'strain': comp_result['strain'],
            'stress': compression_stress,
            'inelastic strain': compression_inelastic_strain,
            'inelastic stress': compression_inelastic_stress,
            'damage': damage_c
        },
        'tension': {
            'crack opening': tension_crack_opening,
            'stress': tension_stress_bilinear,
            'stress exponential': tension_stress_power,
            'cracking strain': tension_cracking_strain,
            'cracking stress': tension_stress_bilinear,
            'damage': damage_t,
            'damage exponential': damage_t_power
        }
    }


def calculate_stress_strain_temp(f_cm, e_c1, e_clim, l_ch):
    """
    Calculate stress-strain relationships for different temperatures.

    Args:
        f_cm: Mean compressive strength [MPa]
        e_c1: Strain at peak compressive strength [-]
        e_clim: Ultimate strain [-]
        l_ch: Characteristic element length [mm]

    Returns:
        dict: Complete results including properties and stress-strain data
    """
    n_points = 40

    # Base material properties
    strength_props = calculate_concrete_strength_properties(f_cm)
    f_ck = strength_props['f_ck']
    f_ctm = strength_props['f_ctm']

    elastic_props = calculate_elastic_modulus(f_cm)
    E_ci = elastic_props['E_ci']
    E_c = elastic_props['E_c']
    E_c1 = f_cm / e_c1

    poisson_props = calculate_poisson_ratios(f_cm, E_c, e_c1)
    v_c0 = poisson_props['v_c0']
    v_ce = poisson_props['v_ce']

    cdp_params = calculate_cdp_parameters(f_cm, E_c, e_c1, v_c0, v_ce)

    G_f = calculate_fracture_energy(f_cm)
    G_c = E_c / (2 * (1 + v_ce))
    l_0 = calculate_characteristic_length(E_c, G_f, f_ctm)

    # Get Eurocode temperature table
    temp_table = get_eurocode_temperature_table()
    temperatures = temp_table[:, 0]

    # Storage for temperature-dependent results
    compression_strain_arrays = []
    compression_stress = []
    compression_inelastic_strain = []
    compression_inelastic_stress = []
    tension_crack_opening = []
    tension_stress_bilinear = []
    tension_stress_power = []
    tension_cracking_strain = []

    # Base properties for temperature calculations
    base_props = {
        'f_cm': f_cm,
        'f_ck': f_ck,
        'f_ctm': f_ctm,
        'E_c1': E_c1
    }

    # Calculate for each temperature
    for temp in temperatures:
        # Apply temperature effects
        temp_props = apply_temperature_effects(base_props, temp, temp_table)
        f_cm_temp = temp_props['f_cm_temp']
        f_ctm_temp = temp_props['f_ctm_temp']
        e_c1_temp = temp_props['e_c1_temp']
        eps_cu = temp_props['eps_cu']
        E_ci_temp = temp_props['E_ci_temp']
        E_c1_temp = temp_props['E_c1_temp']

        print(f'\nTensile Strength [N/mm²], at {temp} [°C]: {f_ctm_temp:.3f}')
        print(f'Tensile Strength [N/mm²] (EC2), at {temp} [°C]: {temp_props["f_ctm_temp_EC"]:.3f}')
        print(f'E-Modul sekant [N/mm²], at {temp} [°C]: {E_c1_temp:.1f}')
        print(f'E-Modul tangent [N/mm²], at {temp} [°C]: {E_ci_temp:.1f}')

        # Compression behavior
        comp_result = calculate_compression_behavior(
            f_cm_temp, e_c1_temp, E_ci_temp, E_c1_temp,
            n_points, eps_cu * 2
        )
        compression_strain_arrays.append(comp_result['strain'])
        compression_stress.append(comp_result['stress'])

        # Inelastic compression
        inel_result = calculate_inelastic_compression(
            comp_result['strain'], comp_result['stress'],
            f_cm_temp, E_c1_temp
        )
        compression_inelastic_strain.append(inel_result['inelastic_strain'])
        compression_inelastic_stress.append(inel_result['inelastic_stress'])

        # Tension behavior
        G_f_temp = calculate_fracture_energy(f_cm_temp)

        # Bilinear tension model
        tension_result = calculate_tension_bilinear(f_ctm_temp, G_f_temp, n_points)
        tension_crack_opening.append(tension_result['crack_opening'])
        tension_stress_bilinear.append(tension_result['stress'])

        # Power law tension model
        stress_power = calculate_tension_power_law(
            f_ctm_temp, G_f_temp,
            tension_result['w_c'],
            tension_result['crack_opening']
        )
        tension_stress_power.append(stress_power)

        # Cracking strain
        cracking_strain = tension_result['crack_opening'] / l_ch
        tension_cracking_strain.append(cracking_strain)

    # Interpolate all curves to match first temperature grid
    for i in range(1, len(temperatures)):
        compression_inelastic_stress[i] = np.interp(
            compression_inelastic_strain[0],
            compression_inelastic_strain[i],
            compression_inelastic_stress[i]
        )
        compression_inelastic_strain[i] = compression_inelastic_strain[0]

        tension_stress_bilinear[i] = np.interp(
            tension_crack_opening[0],
            tension_crack_opening[i],
            tension_stress_bilinear[i]
        )
        tension_stress_power[i] = np.interp(
            tension_crack_opening[0],
            tension_crack_opening[i],
            tension_stress_power[i]
        )
        tension_cracking_strain[i] = tension_cracking_strain[0]
        tension_crack_opening[i] = tension_crack_opening[0]

    # Calculate damage (based on reference temperature)
    damage_c = calculate_compression_damage(
        compression_inelastic_stress[0],
        compression_inelastic_strain[0],
        f_cm
    )

    damage_t = calculate_tension_damage(tension_stress_bilinear[0], f_ctm)
    damage_t_power = calculate_tension_damage(tension_stress_power[0], f_ctm)

    return {
        'properties': {
            'elasticity': E_c,
            'shear': G_c,
            'fracture energy': G_f,
            'tensile strength': f_ctm,
            'dilation angle': cdp_params['dilation_angle'],
            'poisson': v_ce,
            'Kc': cdp_params['K_c'],
            'fbfc': cdp_params['fbfc'],
            'l0': l_0
        },
        'compression': {
            'strain': compression_strain_arrays[0],  # For compatibility
            'strain temp': compression_strain_arrays,  # Temperature-specific strains
            'stress': compression_stress,
            'inelastic strain': compression_inelastic_strain,
            'inelastic stress': compression_inelastic_stress,
            'damage': damage_c
        },
        'tension': {
            'crack opening': tension_crack_opening,
            'stress': tension_stress_bilinear,
            'stress exponential': tension_stress_power,
            'cracking strain': tension_cracking_strain,
            'cracking stress': tension_stress_bilinear,
            'damage': damage_t,
            'damage exponential': damage_t_power
        }
    }


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_curve(x, y, title, xlabel, ylabel, var, style=None):
    """
    Plot a single curve with appropriate labeling.

    Args:
        x: X-axis data
        y: Y-axis data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        var: Variable value (strain rate or temperature)
        style: Optional style dictionary
    """
    if style is None:
        style = {"color": "#1f77b4", "marker": "o"}

    # Determine if strain rate or temperature based on value
    if var < 1:
        label = r'$\dot{\varepsilon}=$' + str(var) + ' [s$^{-1}$]'
    else:
        label = r'$T=$' + str(int(var)) + ' [°C]'

    plt.plot(x, y, label=label, **style, linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_multiple_curves(x, y, title, xlabel, ylabel, var):
    """
    Create a figure with multiple curves.

    Args:
        x: X-axis data (single array or list of arrays)
        y: List of Y-axis data arrays
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        var: List of variable values (strain rates or temperatures)
    """
    custom_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#a55194", "#393b79"
    ]
    custom_markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '<', '>', '8']
    custom_style = cycle([
        {"color": c, "marker": m}
        for c, m in zip(custom_colors, custom_markers)
    ])

    plt.figure()

    # Check if x is a list of arrays or single array
    if len(x) != len(y):
        # Single x-array for all curves
        for i in range(len(y)):
            style = next(custom_style)
            plot_curve(x, y[i], title, xlabel, ylabel, var[i], style)
    else:
        # Different x-array for each curve
        for i in range(len(y)):
            style = next(custom_style)
            plot_curve(x[i], y[i], title, xlabel, ylabel, var[i], style)

    plt.grid()
    plt.legend(loc="upper right")
    plt.show()


# ============================================================================
# Main Program
# ============================================================================

if __name__ == "__main__":

    # Get user inputs with defaults
    f_cm = input("Enter the compressive strength of the concrete (MPa) [Default: 28]: ")
    e_c1 = input("Enter the strain at maximum compressive strength c_i [Default: 0.0022]: ")
    e_clim = input("Enter the strain at ultimate state [Default: 0.0035]: ")
    l_ch = input("Enter the characteristic element length of the mesh (mm) [Default: 1]: ")
    e_rate = input("Enter the strain rates additional to 0/s, separated by a comma [Default: 2,30,100]: ")

    while True:
        temp_input = input("Temperature Dependent Data? (y/n) [Default: n]: ").strip().lower()
        if temp_input in ("y", "n", ""):
            is_strain_rate_mode = (temp_input != "y")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    # Assign default values if no input is provided
    f_cm = float(f_cm.strip()) if f_cm.strip() else 28
    e_c1 = float(e_c1.strip()) if e_c1.strip() else 0.0022
    e_clim = float(e_clim.strip()) if e_clim.strip() else 0.0035
    l_ch = float(l_ch.strip()) if l_ch.strip() else 1
    strain_rates = [0] + list(map(float, e_rate.strip().split(','))) if e_rate.strip() else [0, 2, 30, 100]
    temperatures = np.array([20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100])

    # Calculate stress-strain relationships
    if is_strain_rate_mode:
        results = calculate_stress_strain(f_cm, e_c1, e_clim, l_ch, strain_rates)
        var = strain_rates
        var_label = 'Strain Rate [1/s]'

        # Plot compression
        plot_multiple_curves(
            results['compression']['strain'],
            results['compression']['stress'],
            'Compressive Strain - Compressive Stress',
            'Compressive Strain [-]',
            'Compressive Stress [MPa]',
            var
        )
        plot_multiple_curves(
            results['compression']['inelastic strain'],
            results['compression']['inelastic stress'],
            'Compressive Inelastic Strain - Compressive Stress',
            'Compressive Inelastic Strain [-]',
            'Compressive Stress [MPa]',
            var
        )
    else:
        results = calculate_stress_strain_temp(f_cm, e_c1, e_clim, l_ch)
        var = temperatures
        var_label = 'Temperature [°C]'

        # Plot compression (using temperature-specific strain arrays)
        plot_multiple_curves(
            results['compression']['strain temp'],
            results['compression']['stress'],
            'Compressive Strain - Compressive Stress',
            'Compressive Strain [-]',
            'Compressive Stress [MPa]',
            var
        )
        plot_multiple_curves(
            results['compression']['inelastic strain'],
            results['compression']['inelastic stress'],
            'Compressive Inelastic Strain - Compressive Stress',
            'Compressive Inelastic Strain [-]',
            'Compressive Stress [MPa]',
            var
        )

    # Plot compression damage
    plt.figure()
    plot_curve(
        results['compression']['inelastic strain'][0],
        results['compression']['damage'],
        'Compressive Damage',
        'Compressive Inelastic Strain [-]',
        'Damage [-]',
        var[0]
    )
    plt.grid()
    plt.show()

    # Plot tension
    plot_multiple_curves(
        results['tension']['crack opening'],
        results['tension']['stress'],
        'Crack Opening - Tensile Stress (Bilinear)',
        'Crack Opening [mm]',
        'Cracking Stress [MPa]',
        var
    )
    plot_multiple_curves(
        results['tension']['crack opening'],
        results['tension']['stress exponential'],
        'Crack Opening - Tensile Stress (Power Law)',
        'Crack Opening [mm]',
        'Cracking Stress [MPa]',
        var
    )
    plot_multiple_curves(
        results['tension']['cracking strain'],
        results['tension']['stress'],
        'Cracking Strain - Tensile Stress',
        'Cracking Strain [-]',
        'Cracking Stress [MPa]',
        var
    )
    plot_multiple_curves(
        results['tension']['cracking strain'],
        results['tension']['stress exponential'],
        'Cracking Strain - Tensile Stress (Power Law)',
        'Cracking Strain [-]',
        'Cracking Stress [MPa]',
        var
    )

    # Plot tension damage
    plt.figure()
    plot_curve(
        results['tension']['cracking strain'][0],
        results['tension']['damage'],
        'Tension Damage (Bilinear)',
        'Cracking Strain [-]',
        'Damage [-]',
        var[0]
    )
    plt.grid()
    plt.show()

    plt.figure()
    plot_curve(
        results['tension']['cracking strain'][0],
        results['tension']['damage exponential'],
        'Tension Damage (Power Law)',
        'Cracking Strain [-]',
        'Damage [-]',
        var[0]
    )
    plt.grid()
    plt.show()

    # Print properties
    print('\n' + '='*60)
    print('MATERIAL PROPERTIES SUMMARY')
    print('='*60)
    print(f'Compressive Strength: {f_cm} [MPa]')
    print(f'Tensile Strength: {results["properties"]["tensile strength"]:.2f} [MPa]')
    print(f'Elasticity Modulus: {results["properties"]["elasticity"]:.2f} [MPa]')
    print(f'Poisson: {results["properties"]["poisson"]:.2f} [-]')
    print(f'Shear Modulus: {results["properties"]["shear"]:.2f} [MPa]')
    print(f'Fracture Energy: {results["properties"]["fracture energy"]:.4f} [N/mm]')
    print(f'CDP Dilation angle: {results["properties"]["dilation angle"]:.2f} [°]')
    print(f'CDP fb/fc: {results["properties"]["fbfc"]:.2f} [-]')
    print(f'CDP Kc: {results["properties"]["Kc"]:.2f} [-]')
    print(f'Max. Mesh Size: {results["properties"]["l0"]:.2f} [m]')
    print('='*60)

    # Export to Excel
    print('\nExporting results to Excel...')

    # Prepare export arrays
    var_export_el = np.array([])
    var_export = np.array([])
    compressive_stress_el = np.array([])
    compressive_stress = np.array([])
    compressive_strain_el = np.array([])
    compressive_strain = np.array([])
    var_export_tension = np.array([])
    tension_stress = np.array([])
    tension_strain = np.array([])
    tension_crack = np.array([])
    tension_stress_exp = np.array([])

    for i in range(len(var)):
        # For elastic compression
        if is_strain_rate_mode:
            # Strain rate mode: same strain array for all rates
            strain_length = len(results['compression']['strain'])
            var_export_el = np.concatenate((var_export_el, np.ones(strain_length) * var[i]))
            compressive_strain_el = np.concatenate((compressive_strain_el, results['compression']['strain']))
        else:
            # Temperature mode: different strain arrays
            strain_length = len(results['compression']['strain temp'][i])
            var_export_el = np.concatenate((var_export_el, np.ones(strain_length) * var[i]))
            compressive_strain_el = np.concatenate((compressive_strain_el, results['compression']['strain temp'][i]))

        compressive_stress_el = np.concatenate((compressive_stress_el, results['compression']['stress'][i]))

        # For inelastic compression
        var_export = np.concatenate((var_export, np.ones(len(results['compression']['inelastic strain'][0])) * var[i]))
        compressive_stress = np.concatenate((compressive_stress, results['compression']['inelastic stress'][i]))
        compressive_strain = np.concatenate((compressive_strain, results['compression']['inelastic strain'][i]))

        # For tension
        var_export_tension = np.concatenate((var_export_tension, np.ones(len(results['tension']['cracking strain'][0])) * var[i]))
        tension_stress = np.concatenate((tension_stress, results['tension']['stress'][i]))
        tension_strain = np.concatenate((tension_strain, results['tension']['cracking strain'][i]))
        tension_crack = np.concatenate((tension_crack, results['tension']['crack opening'][i]))
        tension_stress_exp = np.concatenate((tension_stress_exp, results['tension']['stress exponential'][i]))

    # Create DataFrames
    compression_strain_el_df = pd.DataFrame({
        'Compressive Stress [MPa]': compressive_stress_el,
        'Strain [-]': compressive_strain_el,
        var_label: var_export_el
    })

    compression_strain_df = pd.DataFrame({
        'Compressive Stress [MPa]': compressive_stress,
        'Inelastic Strain [-]': compressive_strain,
        var_label: var_export
    })

    compression_damage_df = pd.DataFrame({
        'Damage [-]': results['compression']['damage'],
        'Inelastic Strain [-]': results['compression']['inelastic strain'][0]
    })

    tension_cracking_df = pd.DataFrame({
        'Tension Stress [MPa]': tension_stress,
        'Crack Opening [mm]': tension_crack,
        var_label: var_export_tension
    })

    tension_cracking_strain_df = pd.DataFrame({
        'Tension Stress [MPa]': tension_stress,
        'Cracking Strain [-]': tension_strain,
        var_label: var_export_tension
    })

    tension_damage_df = pd.DataFrame({
        'Damage [-]': results['tension']['damage'],
        'Cracking Strain [-]': results['tension']['cracking strain'][0]
    })

    tension_cracking_power_df = pd.DataFrame({
        'Tension Stress [MPa]': tension_stress_exp,
        'Cracking Strain [-]': tension_strain,
        var_label: var_export_tension
    })

    tension_damage_power_df = pd.DataFrame({
        'Damage [-]': results['tension']['damage exponential'],
        'Cracking Strain [-]': results['tension']['cracking strain'][0]
    })

    # Write to Excel
    excel_file_path = 'CDP-Results.xlsx'
    with pd.ExcelWriter(excel_file_path) as writer:
        compression_strain_el_df.to_excel(writer, sheet_name="Compression Stress-Strain", index=False)
        compression_strain_df.to_excel(writer, sheet_name="Compression Inl.Strain", index=False)
        compression_damage_df.to_excel(writer, sheet_name="Compression Damage", index=False)
        tension_cracking_df.to_excel(writer, sheet_name="Tension Cracking", index=False)
        tension_cracking_strain_df.to_excel(writer, sheet_name="Tension Cr.Strain", index=False)
        tension_damage_df.to_excel(writer, sheet_name="Tension Damage", index=False)
        tension_cracking_power_df.to_excel(writer, sheet_name="Tension Cracking Power", index=False)
        tension_damage_power_df.to_excel(writer, sheet_name="Tension Damage Power", index=False)

    print(f'\nResults exported successfully to {excel_file_path}')
