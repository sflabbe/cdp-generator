"""
Steel material specifications from international standards and calibration.

This module provides:
1. Database of standard steel properties (EC2, ACI, NCh)
2. Functions to retrieve standard specifications
3. Calibration of Johnson-Cook parameters from standard properties

IMPORTANT NOTES:
- Values marked as "APPROXIMATE" should be verified against actual standards
- Standards evolve - always check the latest version for your jurisdiction
- Minimum values are provided; actual material may exceed these
- Override capability is provided for all parameters

Units:
    - Stress: MPa
    - Strain: dimensionless [-]
    - Temperature: °C
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from .johnson_cook import JohnsonCookParams, eng_to_true


@dataclass
class SteelSpec:
    """
    Steel material specification from standards or custom input.

    Attributes:
        standard: Standard code (EC2, ACI, NCh, Custom)
        grade: Steel grade designation (e.g., "B500", "A615_60")
        ductility_class: Ductility class if applicable (A, B, C for EC2)
        fy: Characteristic yield strength [MPa]
        fu: Characteristic ultimate strength [MPa]
        fu_fy_ratio: Ratio fu/fy (alternative to specifying fu directly) [-]
        Agt: Total elongation at maximum force [%]
        E: Elastic modulus [MPa]
        nu: Poisson's ratio [-]
        bar_diameter: Bar diameter [mm] (affects some requirements)
        T_room: Room temperature [°C]
        T_melt: Melting temperature [°C]
        metadata: Additional information (source, notes, etc.)
    """
    standard: str
    grade: str
    ductility_class: Optional[str] = None
    fy: Optional[float] = None  # MPa
    fu: Optional[float] = None  # MPa
    fu_fy_ratio: Optional[float] = None  # dimensionless
    Agt: Optional[float] = None  # %
    E: float = 200000.0  # MPa, typical for steel
    nu: float = 0.30  # dimensionless, typical for steel
    bar_diameter: Optional[float] = None  # mm
    T_room: float = 20.0  # °C
    T_melt: float = 1500.0  # °C, approximate for carbon steel
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate and complete specification."""
        # Calculate fu from ratio if not provided
        if self.fu is None and self.fu_fy_ratio is not None and self.fy is not None:
            self.fu = self.fy * self.fu_fy_ratio
        elif self.fu is not None and self.fy is not None and self.fu_fy_ratio is None:
            self.fu_fy_ratio = self.fu / self.fy

        # Validation
        if self.fy is not None and self.fy <= 0:
            raise ValueError(f"fy must be positive, got {self.fy}")
        if self.fu is not None and self.fu <= 0:
            raise ValueError(f"fu must be positive, got {self.fu}")
        if self.fy is not None and self.fu is not None and self.fu <= self.fy:
            raise ValueError(f"fu ({self.fu}) must be greater than fy ({self.fy})")
        if self.Agt is not None and self.Agt <= 0:
            raise ValueError(f"Agt must be positive, got {self.Agt}")
        if self.E <= 0:
            raise ValueError(f"E must be positive, got {self.E}")


# ============================================================================
# STANDARDS DATABASE
# ============================================================================

# IMPORTANT: These values are APPROXIMATE minimum values from standards.
# Always verify with the actual standard for your application.
# Values are conservative estimates based on common requirements.

STANDARDS_DB = {
    # ========================================================================
    # EUROCODE 2 / EN 10080 - Reinforcing Steel
    # ========================================================================
    "EC2": {
        "B500A": {
            "description": "Eurocode 2 - B500 Class A (Low ductility)",
            "fy": 500.0,  # MPa, characteristic yield
            "fu_fy_ratio": 1.05,  # APPROXIMATE - minimum ratio
            "Agt": 2.5,  # % APPROXIMATE - minimum elongation at max force
            "ductility_class": "A",
            "note": "APPROXIMATE VALUES - Verify with EN 10080 for your region"
        },
        "B500B": {
            "description": "Eurocode 2 - B500 Class B (Normal ductility)",
            "fy": 500.0,  # MPa
            "fu_fy_ratio": 1.08,  # APPROXIMATE
            "Agt": 5.0,  # % APPROXIMATE
            "ductility_class": "B",
            "note": "APPROXIMATE VALUES - Verify with EN 10080 for your region"
        },
        "B500C": {
            "description": "Eurocode 2 - B500 Class C (High ductility)",
            "fy": 500.0,  # MPa
            "fu_fy_ratio": 1.15,  # APPROXIMATE - higher ductility requires higher ratio
            "Agt": 7.5,  # % APPROXIMATE
            "ductility_class": "C",
            "note": "APPROXIMATE VALUES - Verify with EN 10080 for your region"
        },
    },

    # ========================================================================
    # ACI / ASTM - US Standards
    # ========================================================================
    "ACI": {
        "A615_60": {
            "description": "ASTM A615 Grade 60 (Standard rebar)",
            "fy": 414.0,  # MPa (60 ksi)
            "fu": 620.0,  # MPa (90 ksi) APPROXIMATE minimum
            "Agt": 9.0,  # % APPROXIMATE for grade 60
            "note": "APPROXIMATE VALUES - Verify with ASTM A615 for exact requirements"
        },
        "A615_75": {
            "description": "ASTM A615 Grade 75",
            "fy": 517.0,  # MPa (75 ksi)
            "fu": 690.0,  # MPa (100 ksi) APPROXIMATE
            "Agt": 7.0,  # % APPROXIMATE
            "note": "APPROXIMATE VALUES - Verify with ASTM A615"
        },
        "A706_60": {
            "description": "ASTM A706 Grade 60 (Low-alloy, seismic)",
            "fy": 414.0,  # MPa (60 ksi)
            "fu": 550.0,  # MPa APPROXIMATE - A706 has tighter fu/fy control
            "Agt": 14.0,  # % APPROXIMATE - higher ductility for seismic
            "note": "APPROXIMATE VALUES - A706 has specific fu/fy ratio requirements. Verify standard."
        },
    },

    # ========================================================================
    # NCh - Chilean Standard
    # ========================================================================
    "NCh": {
        "A630-420H": {
            "description": "NCh 204 - A630-420H (High ductility rebar)",
            "fy": 420.0,  # MPa
            "fu_fy_ratio": 1.25,  # APPROXIMATE
            "Agt": 8.0,  # % APPROXIMATE
            "note": "APPROXIMATE VALUES - Verify with NCh 204 standard"
        },
        "A440-280H": {
            "description": "NCh 204 - A440-280H",
            "fy": 280.0,  # MPa
            "fu_fy_ratio": 1.20,  # APPROXIMATE
            "Agt": 10.0,  # % APPROXIMATE
            "note": "APPROXIMATE VALUES - Verify with NCh 204 standard"
        },
    },
}


def get_steel_spec(
    standard: str,
    grade: str,
    ductility_class: Optional[str] = None,
    bar_diameter: Optional[float] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> SteelSpec:
    """
    Retrieve steel specification from standards database.

    Args:
        standard: Standard code (EC2, ACI, NCh, Custom)
        grade: Steel grade (e.g., "B500B", "A615_60")
        ductility_class: Ductility class (A/B/C) if not in grade name
        bar_diameter: Bar diameter [mm] (some standards have diameter-dependent values)
        overrides: Dictionary of properties to override (fy, fu, Agt, E, etc.)

    Returns:
        SteelSpec object with complete specification

    Raises:
        ValueError: If standard or grade not found, or if insufficient data

    Examples:
        >>> spec = get_steel_spec("EC2", "B500C")
        >>> spec = get_steel_spec("ACI", "A615_60")
        >>> spec = get_steel_spec("EC2", "B500B", overrides={"fy": 550, "Agt": 6.0})
        >>> spec = get_steel_spec("Custom", "MySteel", overrides={"fy": 500, "fu": 600, "Agt": 10})
    """
    # Normalize standard name (case-insensitive matching)
    standard_normalized = standard.upper() if standard.upper() == "CUSTOM" else standard

    # Check if standard exists (case-sensitive for database keys)
    if standard_normalized not in STANDARDS_DB and standard.upper() != "CUSTOM":
        available = list(STANDARDS_DB.keys()) + ["CUSTOM"]
        raise ValueError(
            f"Standard '{standard}' not found. Available: {available}"
        )

    # Handle custom specification
    if standard.upper() == "CUSTOM":
        if overrides is None:
            raise ValueError(
                "Custom standard requires overrides with at least: fy, fu (or fu_fy_ratio), Agt"
            )

        # Build custom spec
        spec_data = {
            "standard": "Custom",
            "grade": grade,
            "ductility_class": ductility_class,
            "bar_diameter": bar_diameter,
        }
        spec_data.update(overrides)

        spec = SteelSpec(**spec_data)

        # Validate custom has minimum required data
        if spec.fy is None:
            raise ValueError("Custom standard requires 'fy' in overrides")
        if spec.fu is None and spec.fu_fy_ratio is None:
            raise ValueError("Custom standard requires 'fu' or 'fu_fy_ratio' in overrides")
        if spec.Agt is None:
            raise ValueError("Custom standard requires 'Agt' in overrides")

        return spec

    # Retrieve from database
    if grade not in STANDARDS_DB[standard]:
        available_grades = list(STANDARDS_DB[standard].keys())
        raise ValueError(
            f"Grade '{grade}' not found in {standard}. Available: {available_grades}"
        )

    db_entry = STANDARDS_DB[standard][grade]

    # Build spec from database
    spec_data = {
        "standard": standard,
        "grade": grade,
        "fy": db_entry.get("fy"),
        "fu": db_entry.get("fu"),
        "fu_fy_ratio": db_entry.get("fu_fy_ratio"),
        "Agt": db_entry.get("Agt"),
        "ductility_class": db_entry.get("ductility_class", ductility_class),
        "bar_diameter": bar_diameter,
        "metadata": {
            "description": db_entry.get("description", ""),
            "note": db_entry.get("note", ""),
            "source": "Built-in standards database (APPROXIMATE VALUES)"
        }
    }

    # Apply overrides
    if overrides is not None:
        spec_data.update(overrides)

    return SteelSpec(**spec_data)


# ============================================================================
# JOHNSON-COOK CALIBRATION FROM STANDARDS
# ============================================================================

def calibrate_jc_from_spec(
    spec: SteelSpec,
    *,
    assume_rate_temp_neutral: bool = True,
    n_default: Optional[float] = None,
    C_default: float = 0.0,
    m_default: float = 0.0,
    epsdot0: float = 1e-3,
    verbose: bool = False
) -> JohnsonCookParams:
    """
    Calibrate Johnson-Cook parameters from steel specification.

    Strategy:
    1. Use fy as parameter A (yield stress at reference conditions)
    2. Use fu and Agt to determine ultimate point in true stress-strain
    3. Fit B and n to match the ultimate point
    4. Set C and m based on assumptions or defaults

    Args:
        spec: SteelSpec with at least fy, fu, Agt, E
        assume_rate_temp_neutral: If True, set C=0 and m=0 (default: True)
        n_default: Default strain hardening exponent. If None, estimate from ductility class
        C_default: Default strain rate coefficient (default: 0.0)
        m_default: Default thermal softening exponent (default: 0.0)
        epsdot0: Reference strain rate [1/s] (default: 1e-3)
        verbose: Print calibration details

    Returns:
        JohnsonCookParams calibrated from spec

    Raises:
        ValueError: If spec lacks required data or calibration fails

    Notes:
        - Calibration assumes quasi-static loading at room temperature
        - n_default heuristics (if not provided):
            * Class C (high ductility): n = 0.20
            * Class B (normal ductility): n = 0.15
            * Class A (low ductility): n = 0.10
            * Unknown: n = 0.15
        - For better accuracy, provide experimental stress-strain data and use curve fitting
    """
    # Validate required data
    if spec.fy is None:
        raise ValueError("SteelSpec must have 'fy' defined")
    if spec.fu is None:
        raise ValueError("SteelSpec must have 'fu' or 'fu_fy_ratio' defined")
    if spec.Agt is None:
        raise ValueError("SteelSpec must have 'Agt' (elongation at max force) defined")
    if spec.E is None or spec.E <= 0:
        raise ValueError("SteelSpec must have valid elastic modulus 'E'")

    # Parameter A: yield stress
    A = spec.fy

    # Determine n (strain hardening exponent)
    if n_default is None:
        # Heuristic based on ductility class
        if spec.ductility_class is not None:
            dc = spec.ductility_class.upper()
            if dc == "C":
                n = 0.20  # High ductility
            elif dc == "B":
                n = 0.15  # Normal ductility
            elif dc == "A":
                n = 0.10  # Low ductility
            else:
                n = 0.15  # Default
        else:
            # Estimate from fu/fy ratio if available
            if spec.fu_fy_ratio is not None:
                if spec.fu_fy_ratio >= 1.20:
                    n = 0.18
                elif spec.fu_fy_ratio >= 1.10:
                    n = 0.15
                else:
                    n = 0.12
            else:
                n = 0.15  # Default
    else:
        n = n_default

    # Calculate ultimate point in engineering stress-strain
    eps_u_eng = spec.Agt / 100.0  # Convert % to fraction
    sigma_u_eng = spec.fu

    # Convert to true stress-strain
    eps_u_true_arr, sigma_u_true_arr = eng_to_true(eps_u_eng, sigma_u_eng)
    eps_u_true = float(eps_u_true_arr[0]) if hasattr(eps_u_true_arr, '__len__') else float(eps_u_true_arr)
    sigma_u_true = float(sigma_u_true_arr[0]) if hasattr(sigma_u_true_arr, '__len__') else float(sigma_u_true_arr)

    # Calculate plastic strain at ultimate point
    eps_p_u = eps_u_true - sigma_u_true / spec.E

    if eps_p_u <= 0:
        raise ValueError(
            f"Calculated plastic strain at ultimate is non-positive ({eps_p_u:.6f}). "
            f"Check that fu > fy and Agt is reasonable. "
            f"(fy={spec.fy}, fu={spec.fu}, Agt={spec.Agt}%)"
        )

    # Fit B to match ultimate stress
    # At ultimate: sigma_u_true = A + B * eps_p_u^n
    # Therefore: B = (sigma_u_true - A) / eps_p_u^n

    if sigma_u_true <= A:
        raise ValueError(
            f"Ultimate true stress ({sigma_u_true:.2f} MPa) must be greater than yield (A={A:.2f} MPa). "
            f"Check fu/fy ratio and Agt values."
        )

    B = float((sigma_u_true - A) / np.power(eps_p_u, n))

    if B < 0:
        raise ValueError(
            f"Calculated negative B ({B:.2f}). This indicates inconsistent input data."
        )

    # Rate and temperature parameters
    if assume_rate_temp_neutral:
        C = 0.0
        m = 0.0
    else:
        C = C_default
        m = m_default

    # Create JC parameters
    params = JohnsonCookParams(
        A=A,
        B=B,
        n=n,
        C=C,
        m=m,
        epsdot0=epsdot0,
        T_room=spec.T_room,
        T_melt=spec.T_melt
    )

    if verbose:
        print("=" * 70)
        print("JOHNSON-COOK CALIBRATION SUMMARY")
        print("=" * 70)
        print(f"Standard: {spec.standard} {spec.grade}")
        if spec.ductility_class:
            print(f"Ductility class: {spec.ductility_class}")
        print(f"\nInput properties:")
        print(f"  fy = {spec.fy:.2f} MPa")
        print(f"  fu = {spec.fu:.2f} MPa")
        print(f"  fu/fy = {spec.fu_fy_ratio:.3f}")
        print(f"  Agt = {spec.Agt:.2f} %")
        print(f"  E = {spec.E:.0f} MPa")
        print(f"\nCalculated ultimate point:")
        print(f"  Engineering: ε_u = {eps_u_eng:.6f}, σ_u = {sigma_u_eng:.2f} MPa")
        print(f"  True: ε_u = {eps_u_true:.6f}, σ_u = {sigma_u_true:.2f} MPa")
        print(f"  Plastic: ε_p_u = {eps_p_u:.6f}")
        print(f"\nCalibrated Johnson-Cook parameters:")
        print(f"  A = {params.A:.2f} MPa  (yield stress)")
        print(f"  B = {params.B:.2f} MPa  (hardening coefficient)")
        print(f"  n = {params.n:.4f}  (hardening exponent)")
        print(f"  C = {params.C:.4f}  (rate sensitivity)")
        print(f"  m = {params.m:.4f}  (thermal softening)")
        print(f"  ε̇₀ = {params.epsdot0:.2e} 1/s")
        print(f"  T_room = {params.T_room:.1f} °C")
        print(f"  T_melt = {params.T_melt:.1f} °C")
        print("=" * 70)

        # Verification
        from .johnson_cook import johnson_cook_flow_stress
        sigma_check = johnson_cook_flow_stress(
            eps_p_u, epsdot0, spec.T_room, params
        )[0]
        error_pct = 100.0 * abs(sigma_check - sigma_u_true) / sigma_u_true
        print(f"\nVerification:")
        print(f"  Target σ(ε_p_u) = {sigma_u_true:.2f} MPa")
        print(f"  JC model σ(ε_p_u) = {sigma_check:.2f} MPa")
        print(f"  Error = {error_pct:.2f} %")
        print("=" * 70)

    return params


def list_available_standards() -> Dict[str, list]:
    """
    List all available standards and grades.

    Returns:
        Dictionary with standard names as keys and lists of grades as values
    """
    result = {}
    for standard, grades_dict in STANDARDS_DB.items():
        result[standard] = list(grades_dict.keys())
    return result


def print_standards_info():
    """Print information about all available standards."""
    print("=" * 70)
    print("AVAILABLE STEEL STANDARDS")
    print("=" * 70)
    print("\nIMPORTANT: Values are APPROXIMATE. Always verify with actual standards.")
    print()

    for standard, grades_dict in STANDARDS_DB.items():
        print(f"\n{standard}:")
        print("-" * 70)
        for grade, data in grades_dict.items():
            print(f"\n  {grade}:")
            print(f"    Description: {data.get('description', 'N/A')}")
            if 'fy' in data:
                print(f"    fy = {data['fy']:.0f} MPa")
            if 'fu' in data:
                print(f"    fu = {data['fu']:.0f} MPa")
            if 'fu_fy_ratio' in data:
                print(f"    fu/fy = {data['fu_fy_ratio']:.2f}")
            if 'Agt' in data:
                print(f"    Agt = {data['Agt']:.1f} %")
            if 'ductility_class' in data:
                print(f"    Ductility class: {data['ductility_class']}")
            if 'note' in data:
                print(f"    NOTE: {data['note']}")

    print("\n" + "=" * 70)
    print("\nTo use custom values:")
    print("  standard='Custom', grade='YourGrade',")
    print("  overrides={'fy': 500, 'fu': 600, 'Agt': 10}")
    print("=" * 70)
