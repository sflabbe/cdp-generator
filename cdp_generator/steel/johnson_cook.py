"""
Johnson-Cook constitutive model for steel materials.

This module implements the Johnson-Cook (JC) plasticity model, commonly used
for modeling the behavior of metals under large strains, high strain rates,
and elevated temperatures.

The Johnson-Cook flow stress equation is:
    σ = [A + B * εₚⁿ] * [1 + C * ln(ε̇/ε̇₀)] * [1 - T*ᵐ]

where:
    - εₚ: equivalent plastic strain [-]
    - ε̇: strain rate [1/s]
    - T*: homologous temperature, (T - T_room)/(T_melt - T_room), clamped to [0,1]
    - A, B, n, C, m: material parameters
    - ε̇₀: reference strain rate [1/s]

Units:
    - Stress: MPa
    - Strain: dimensionless [-]
    - Strain rate: 1/s
    - Temperature: °C (converted internally to K where needed)
    - Elastic modulus: MPa
"""

from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np


@dataclass
class JohnsonCookParams:
    """
    Parameters for the Johnson-Cook constitutive model.

    Attributes:
        A: Yield stress at reference conditions [MPa]
        B: Strain hardening coefficient [MPa]
        n: Strain hardening exponent [-]
        C: Strain rate sensitivity coefficient [-]
        m: Thermal softening exponent [-]
        epsdot0: Reference strain rate [1/s]
        T_room: Room temperature [°C]
        T_melt: Melting temperature [°C]

    The flow stress equation is:
        σ = [A + B * εₚⁿ] * [1 + C * ln(ε̇/ε̇₀)] * [1 - T*ᵐ]

    where T* = (T - T_room)/(T_melt - T_room), clamped to [0, 1]
    """
    A: float  # MPa
    B: float  # MPa
    n: float  # dimensionless
    C: float  # dimensionless
    m: float  # dimensionless
    epsdot0: float  # 1/s
    T_room: float  # °C
    T_melt: float  # °C

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.A <= 0:
            raise ValueError(f"A must be positive, got {self.A}")
        if self.B < 0:
            raise ValueError(f"B must be non-negative, got {self.B}")
        if self.n < 0:
            raise ValueError(f"n must be non-negative, got {self.n}")
        if self.epsdot0 <= 0:
            raise ValueError(f"epsdot0 must be positive, got {self.epsdot0}")
        if self.T_melt <= self.T_room:
            raise ValueError(
                f"T_melt ({self.T_melt}) must be greater than T_room ({self.T_room})"
            )


def johnson_cook_flow_stress(
    eps_p: Union[float, np.ndarray],
    epsdot: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    params: JohnsonCookParams
) -> np.ndarray:
    """
    Calculate flow stress using the Johnson-Cook model.

    σ = [A + B * εₚⁿ] * [1 + C * ln(ε̇/ε̇₀)] * [1 - T*ᵐ]

    where T* = (T - T_room)/(T_melt - T_room), clamped to [0, 1]

    Args:
        eps_p: Equivalent plastic strain [-], scalar or array
        epsdot: Strain rate [1/s], scalar or array
        T: Temperature [°C], scalar or array
        params: Johnson-Cook parameters

    Returns:
        Flow stress [MPa] as numpy array

    Notes:
        - All inputs are automatically converted to numpy arrays
        - Negative plastic strains are clipped to 0
        - Zero or negative strain rates are replaced with epsdot0 to avoid log errors
        - Homologous temperature T* is clamped to [0, 1]
        - Result is clipped to non-negative values
    """
    # Convert to arrays for consistent handling
    eps_p = np.atleast_1d(eps_p).astype(float)
    epsdot = np.atleast_1d(epsdot).astype(float)
    T = np.atleast_1d(T).astype(float)

    # Clip plastic strain to non-negative
    eps_p = np.maximum(eps_p, 0.0)

    # Handle strain rate safely (avoid log of zero or negative)
    epsdot_safe = np.where(epsdot <= 0, params.epsdot0, epsdot)

    # Calculate homologous temperature T* = (T - T_room) / (T_melt - T_room)
    T_star = (T - params.T_room) / (params.T_melt - params.T_room)
    T_star = np.clip(T_star, 0.0, 1.0)

    # Johnson-Cook equation components
    hardening_term = params.A + params.B * np.power(eps_p, params.n)
    rate_term = 1.0 + params.C * np.log(epsdot_safe / params.epsdot0)

    # Handle thermal term carefully: when m=0, thermal effect is disabled
    if params.m == 0:
        thermal_term = np.ones_like(T_star)
    else:
        thermal_term = 1.0 - np.power(T_star, params.m)

    # Combined flow stress
    sigma = hardening_term * rate_term * thermal_term

    # Clip to non-negative (can happen at very high temperatures)
    sigma = np.maximum(sigma, 0.0)

    return sigma


def eng_to_true(
    eps_eng: Union[float, np.ndarray],
    sigma_eng: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert engineering strain and stress to true (logarithmic) values.

    Formulas:
        ε_true = ln(1 + ε_eng)
        σ_true = σ_eng * (1 + ε_eng)

    Args:
        eps_eng: Engineering strain [-]
        sigma_eng: Engineering stress [MPa]

    Returns:
        Tuple of (eps_true, sigma_true) as numpy arrays

    Notes:
        - Assumes volume conservation (valid for metals in plastic regime)
        - Engineering strain should be non-negative for valid conversion
    """
    eps_eng = np.atleast_1d(eps_eng).astype(float)
    sigma_eng = np.atleast_1d(sigma_eng).astype(float)

    eps_true = np.log(1.0 + eps_eng)
    sigma_true = sigma_eng * (1.0 + eps_eng)

    return eps_true, sigma_true


def true_to_eng(
    eps_true: Union[float, np.ndarray],
    sigma_true: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert true (logarithmic) strain and stress to engineering values.

    Formulas:
        ε_eng = exp(ε_true) - 1
        σ_eng = σ_true / (1 + ε_eng) = σ_true / exp(ε_true)

    Args:
        eps_true: True strain [-]
        sigma_true: True stress [MPa]

    Returns:
        Tuple of (eps_eng, sigma_eng) as numpy arrays
    """
    eps_true = np.atleast_1d(eps_true).astype(float)
    sigma_true = np.atleast_1d(sigma_true).astype(float)

    eps_eng = np.exp(eps_true) - 1.0
    sigma_eng = sigma_true / np.exp(eps_true)

    return eps_eng, sigma_eng


def true_plastic_strain(
    eps_true: Union[float, np.ndarray],
    sigma_true: Union[float, np.ndarray],
    E: float
) -> np.ndarray:
    """
    Calculate plastic component of true strain.

    Formula:
        εₚ = ε_true - σ_true / E

    Args:
        eps_true: True total strain [-]
        sigma_true: True stress [MPa]
        E: Elastic modulus [MPa]

    Returns:
        Plastic strain [-] as numpy array

    Notes:
        - Plastic strain is clipped to non-negative values
        - E must be positive
    """
    if E <= 0:
        raise ValueError(f"Elastic modulus E must be positive, got {E}")

    eps_true = np.atleast_1d(eps_true).astype(float)
    sigma_true = np.atleast_1d(sigma_true).astype(float)

    eps_p = eps_true - sigma_true / E
    eps_p = np.maximum(eps_p, 0.0)

    return eps_p


def generate_jc_curve(
    params: JohnsonCookParams,
    E: float,
    eps_max: float = 0.20,
    n_points: int = 100,
    epsdot: float = 1e-3,
    T: float = 20.0,
    output_kind: str = "true"
) -> dict:
    """
    Generate a complete Johnson-Cook stress-strain curve.

    Args:
        params: Johnson-Cook parameters
        E: Elastic modulus [MPa]
        eps_max: Maximum total strain [-] (default: 0.20 = 20%)
        n_points: Number of points in the curve (default: 100)
        epsdot: Strain rate [1/s] (default: 1e-3 = quasi-static)
        T: Temperature [°C] (default: 20)
        output_kind: "true" for true strain/stress, "engineering" for engineering (default: "true")

    Returns:
        Dictionary with:
            - 'strain': Total strain array [-]
            - 'stress': Stress array [MPa]
            - 'plastic_strain': Plastic strain array [-]
            - 'elastic_strain': Elastic strain array [-]
            - 'output_kind': "true" or "engineering"
            - 'epsdot': Strain rate used [1/s]
            - 'T': Temperature used [°C]

    Notes:
        - Curve is generated in true stress-strain space
        - If output_kind="engineering", result is converted
        - Elastic-plastic transition is handled smoothly
    """
    if output_kind not in ["true", "engineering"]:
        raise ValueError(f"output_kind must be 'true' or 'engineering', got '{output_kind}'")

    if E <= 0:
        raise ValueError(f"E must be positive, got {E}")

    if eps_max <= 0:
        raise ValueError(f"eps_max must be positive, got {eps_max}")

    # Generate true strain array
    eps_true = np.linspace(0, eps_max, n_points)

    # Calculate yield point in true strain
    sigma_y_true = params.A  # At eps_p=0, epsdot=epsdot0, T=T_room
    eps_y_true = sigma_y_true / E

    # Initialize arrays
    sigma_true = np.zeros_like(eps_true)
    eps_p = np.zeros_like(eps_true)

    for i, eps in enumerate(eps_true):
        if eps <= eps_y_true:
            # Elastic region
            sigma_true[i] = E * eps
            eps_p[i] = 0.0
        else:
            # Plastic region: iterate to find consistent eps_p
            # Initial guess
            eps_p_guess = eps - sigma_y_true / E

            # Simple fixed-point iteration
            for _ in range(10):
                sigma_guess = johnson_cook_flow_stress(eps_p_guess, epsdot, T, params)[0]
                eps_p_new = eps - sigma_guess / E
                if abs(eps_p_new - eps_p_guess) < 1e-8:
                    break
                eps_p_guess = eps_p_new

            eps_p[i] = eps_p_guess
            sigma_true[i] = johnson_cook_flow_stress(eps_p[i], epsdot, T, params)[0]

    eps_elastic = eps_true - eps_p

    # Convert to engineering if requested
    if output_kind == "engineering":
        eps_eng, sigma_eng = true_to_eng(eps_true, sigma_true)
        _, sigma_eng_check = true_to_eng(eps_true, sigma_true)

        return {
            'strain': eps_eng,
            'stress': sigma_eng,
            'plastic_strain': eps_p,  # Keep plastic in true strain (more meaningful)
            'elastic_strain': eps_elastic,
            'output_kind': 'engineering',
            'epsdot': epsdot,
            'T': T
        }
    else:
        return {
            'strain': eps_true,
            'stress': sigma_true,
            'plastic_strain': eps_p,
            'elastic_strain': eps_elastic,
            'output_kind': 'true',
            'epsdot': epsdot,
            'T': T
        }


def generate_jc_curves_multicase(
    params: JohnsonCookParams,
    E: float,
    eps_max: float = 0.20,
    n_points: int = 100,
    strain_rates: Optional[list] = None,
    temperatures: Optional[list] = None,
    output_kind: str = "true"
) -> dict:
    """
    Generate Johnson-Cook curves for multiple strain rates and/or temperatures.

    Args:
        params: Johnson-Cook parameters
        E: Elastic modulus [MPa]
        eps_max: Maximum total strain [-]
        n_points: Number of points per curve
        strain_rates: List of strain rates [1/s] (default: [1e-3])
        temperatures: List of temperatures [°C] (default: [20])
        output_kind: "true" or "engineering"

    Returns:
        Dictionary with:
            - 'curves': List of curve dicts (one per rate-temp combination)
            - 'strain_rates': Array of strain rates used
            - 'temperatures': Array of temperatures used
            - 'params': JohnsonCookParams used
            - 'E': Elastic modulus used
    """
    if strain_rates is None:
        strain_rates = [1e-3]
    if temperatures is None:
        temperatures = [20.0]

    curves = []

    for epsdot in strain_rates:
        for T in temperatures:
            curve = generate_jc_curve(
                params=params,
                E=E,
                eps_max=eps_max,
                n_points=n_points,
                epsdot=epsdot,
                T=T,
                output_kind=output_kind
            )
            curve['case_id'] = f"rate={epsdot:.2e}_T={T:.1f}"
            curves.append(curve)

    return {
        'curves': curves,
        'strain_rates': np.array(strain_rates),
        'temperatures': np.array(temperatures),
        'params': params,
        'E': E
    }
