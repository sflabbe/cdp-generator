"""
Tension Behavior Module

Functions for calculating tension behavior and damage.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def calculate_tension_bilinear(f_ctm: float, G_f: float, n_points: int) -> dict[str, Any]:
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

    return {"crack_opening": crack_opening, "stress": stress, "w_1": w_1, "w_c": w_c}


def calculate_tension_power_law(
    f_ctm: float, G_f: float, w_c: float, crack_opening: NDArray[np.float64]
) -> NDArray[np.float64]:
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


def calculate_tension_damage(stress: NDArray[np.float64], f_ctm: float) -> NDArray[np.float64]:
    """
    Calculate tension damage parameter.

    Args:
        stress: Stress array [MPa]
        f_ctm: Tensile strength [MPa]

    Returns:
        ndarray: Damage array [-]
    """
    return 1 - stress / f_ctm
