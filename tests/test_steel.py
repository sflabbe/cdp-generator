"""
Tests for steel Johnson-Cook module.

These tests verify:
1. JC flow stress increases monotonically with plastic strain
2. Calibration hits target fu
3. Vectorization works correctly
4. Basic sanity checks
"""

import pytest
import numpy as np

from cdp_generator.steel import (
    JohnsonCookParams,
    johnson_cook_flow_stress,
    eng_to_true,
    true_to_eng,
    true_plastic_strain,
    generate_jc_curve,
    generate_jc_curves_multicase,
    get_steel_spec,
    calibrate_jc_from_spec,
    SteelSpec,
)


# ============================================================================
# Test JohnsonCookParams validation
# ============================================================================

def test_jc_params_validation():
    """Test that JohnsonCookParams validates inputs."""
    # Valid params should work
    params = JohnsonCookParams(
        A=500, B=300, n=0.15, C=0.01, m=1.0,
        epsdot0=1e-3, T_room=20, T_melt=1500
    )
    assert params.A == 500

    # Negative A should fail
    with pytest.raises(ValueError):
        JohnsonCookParams(
            A=-500, B=300, n=0.15, C=0.01, m=1.0,
            epsdot0=1e-3, T_room=20, T_melt=1500
        )

    # T_melt <= T_room should fail
    with pytest.raises(ValueError):
        JohnsonCookParams(
            A=500, B=300, n=0.15, C=0.01, m=1.0,
            epsdot0=1e-3, T_room=1500, T_melt=1500
        )


# ============================================================================
# Test JC flow stress - monotonic behavior
# ============================================================================

def test_jc_flow_stress_monotonic():
    """Test that flow stress increases monotonically with plastic strain."""
    params = JohnsonCookParams(
        A=500, B=300, n=0.15, C=0.0, m=0.0,
        epsdot0=1e-3, T_room=20, T_melt=1500
    )

    # Test at room temperature and reference strain rate
    eps_p = np.linspace(0, 0.20, 50)
    sigma = johnson_cook_flow_stress(eps_p, 1e-3, 20, params)

    # Check monotonically increasing
    assert np.all(np.diff(sigma) >= 0), "Stress should increase monotonically with plastic strain"

    # Check initial value is A
    assert np.isclose(sigma[0], params.A, rtol=1e-3)

    # Check final value is A + B*eps_p_max^n
    sigma_expected = params.A + params.B * np.power(eps_p[-1], params.n)
    assert np.isclose(sigma[-1], sigma_expected, rtol=1e-3)


# ============================================================================
# Test strain rate effect
# ============================================================================

def test_strain_rate_effect():
    """Test that higher strain rate increases flow stress (when C > 0)."""
    params = JohnsonCookParams(
        A=500, B=300, n=0.15, C=0.01, m=0.0,
        epsdot0=1e-3, T_room=20, T_melt=1500
    )

    eps_p = 0.10
    T = 20

    # Calculate at different strain rates
    sigma_low = johnson_cook_flow_stress(eps_p, 1e-3, T, params)[0]
    sigma_high = johnson_cook_flow_stress(eps_p, 100, T, params)[0]

    # Higher rate should give higher stress
    assert sigma_high > sigma_low, "Higher strain rate should increase stress"


# ============================================================================
# Test temperature effect
# ============================================================================

def test_temperature_effect():
    """Test that higher temperature decreases flow stress (when m > 0)."""
    params = JohnsonCookParams(
        A=500, B=300, n=0.15, C=0.0, m=1.0,
        epsdot0=1e-3, T_room=20, T_melt=1500
    )

    eps_p = 0.10
    epsdot = 1e-3

    # Calculate at different temperatures
    sigma_low_T = johnson_cook_flow_stress(eps_p, epsdot, 20, params)[0]
    sigma_high_T = johnson_cook_flow_stress(eps_p, epsdot, 800, params)[0]

    # Higher temperature should give lower stress
    assert sigma_high_T < sigma_low_T, "Higher temperature should decrease stress"


# ============================================================================
# Test vectorization
# ============================================================================

def test_vectorization():
    """Test that inputs can be arrays and outputs are arrays."""
    params = JohnsonCookParams(
        A=500, B=300, n=0.15, C=0.0, m=0.0,
        epsdot0=1e-3, T_room=20, T_melt=1500
    )

    # Array inputs
    eps_p = np.array([0, 0.05, 0.10, 0.15])
    epsdot = np.array([1e-3, 1e-2, 1e-1, 1.0])
    T = np.array([20, 100, 200, 300])

    sigma = johnson_cook_flow_stress(eps_p, epsdot, T, params)

    assert isinstance(sigma, np.ndarray)
    assert sigma.shape == eps_p.shape


# ============================================================================
# Test engineering-true conversions
# ============================================================================

def test_eng_true_conversions():
    """Test engineering to true strain/stress conversions."""
    eps_eng = 0.10
    sigma_eng = 600

    # Convert to true
    eps_true, sigma_true = eng_to_true(eps_eng, sigma_eng)

    # Check formulas
    assert np.isclose(eps_true, np.log(1 + eps_eng))
    assert np.isclose(sigma_true, sigma_eng * (1 + eps_eng))

    # Convert back
    eps_eng_back, sigma_eng_back = true_to_eng(eps_true, sigma_true)

    # Should match original
    assert np.isclose(eps_eng_back, eps_eng, rtol=1e-6)
    assert np.isclose(sigma_eng_back, sigma_eng, rtol=1e-6)


def test_true_plastic_strain():
    """Test plastic strain calculation."""
    E = 200000  # MPa
    eps_true = 0.10
    sigma_true = 700

    eps_p = true_plastic_strain(eps_true, sigma_true, E)

    # Should be total - elastic
    expected = eps_true - sigma_true / E
    assert np.isclose(eps_p, expected)

    # Should be non-negative
    assert eps_p >= 0


# ============================================================================
# Test calibration from standard specs
# ============================================================================

def test_calibration_hits_fu():
    """Test that calibrated JC parameters reproduce the ultimate stress."""
    # Get a standard spec
    spec = get_steel_spec("EC2", "B500C")

    # Calibrate
    params = calibrate_jc_from_spec(spec, verbose=False)

    # Calculate ultimate point
    eps_u_eng = spec.Agt / 100.0
    sigma_u_eng = spec.fu
    eps_u_true, sigma_u_true = eng_to_true(eps_u_eng, sigma_u_eng)
    eps_p_u = true_plastic_strain(eps_u_true, sigma_u_true, spec.E)

    # Evaluate JC model at ultimate point
    sigma_jc_array = johnson_cook_flow_stress(eps_p_u, params.epsdot0, params.T_room, params)
    sigma_jc = float(sigma_jc_array[0])

    # Should match within 3%
    error_pct = 100.0 * abs(sigma_jc - sigma_u_true) / sigma_u_true
    assert error_pct < 3.0, f"JC model error at ultimate point: {error_pct:.2f}%"


def test_calibration_yield_stress():
    """Test that calibrated parameter A matches yield stress."""
    spec = get_steel_spec("ACI", "A615_60")
    params = calibrate_jc_from_spec(spec, verbose=False)

    # A should equal fy
    assert np.isclose(params.A, spec.fy, rtol=1e-6)


def test_get_steel_spec():
    """Test retrieving steel specifications from database."""
    # Test EC2
    spec_ec2 = get_steel_spec("EC2", "B500B")
    assert spec_ec2.fy == 500
    assert spec_ec2.ductility_class == "B"

    # Test ACI
    spec_aci = get_steel_spec("ACI", "A615_60")
    assert spec_aci.fy == 414  # 60 ksi in MPa

    # Test NCh
    spec_nch = get_steel_spec("NCh", "A630-420H")
    assert spec_nch.fy == 420

    # Test with overrides
    spec_override = get_steel_spec("EC2", "B500C", overrides={"fy": 550})
    assert spec_override.fy == 550

    # Test custom
    spec_custom = get_steel_spec(
        "Custom", "MySteel",
        overrides={"fy": 500, "fu": 600, "Agt": 10}
    )
    assert spec_custom.fy == 500
    assert spec_custom.fu == 600


# ============================================================================
# Test curve generation
# ============================================================================

def test_generate_jc_curve():
    """Test generating a single JC curve."""
    params = JohnsonCookParams(
        A=500, B=300, n=0.15, C=0.0, m=0.0,
        epsdot0=1e-3, T_room=20, T_melt=1500
    )

    curve = generate_jc_curve(
        params=params,
        E=200000,
        eps_max=0.20,
        n_points=100,
        epsdot=1e-3,
        T=20,
        output_kind="true"
    )

    # Check structure
    assert 'strain' in curve
    assert 'stress' in curve
    assert 'plastic_strain' in curve
    assert 'elastic_strain' in curve
    assert curve['output_kind'] == 'true'

    # Check arrays have correct length
    assert len(curve['strain']) == 100
    assert len(curve['stress']) == 100

    # Check monotonicity
    assert np.all(np.diff(curve['stress']) >= 0)

    # Check initial elastic behavior (stress = E * strain)
    # Exclude strain=0 to avoid division by zero
    idx_elastic = np.where((curve['plastic_strain'] < 1e-6) & (curve['strain'] > 1e-8))[0]
    if len(idx_elastic) > 1:
        elastic_modulus = curve['stress'][idx_elastic] / curve['strain'][idx_elastic]
        assert np.allclose(elastic_modulus, 200000, rtol=0.01)


def test_generate_jc_curves_multicase():
    """Test generating multiple JC curves."""
    params = JohnsonCookParams(
        A=500, B=300, n=0.15, C=0.01, m=1.0,
        epsdot0=1e-3, T_room=20, T_melt=1500
    )

    results = generate_jc_curves_multicase(
        params=params,
        E=200000,
        strain_rates=[1e-3, 1, 100],
        temperatures=[20, 400, 800],
        n_points=50
    )

    # Should have 3 x 3 = 9 curves
    assert len(results['curves']) == 9

    # Check structure
    assert 'strain_rates' in results
    assert 'temperatures' in results
    assert len(results['strain_rates']) == 3
    assert len(results['temperatures']) == 3


# ============================================================================
# Test edge cases
# ============================================================================

def test_zero_plastic_strain():
    """Test that zero plastic strain gives yield stress."""
    params = JohnsonCookParams(
        A=500, B=300, n=0.15, C=0.0, m=0.0,
        epsdot0=1e-3, T_room=20, T_melt=1500
    )

    sigma = johnson_cook_flow_stress(0.0, 1e-3, 20, params)[0]
    assert np.isclose(sigma, params.A)


def test_high_temperature_clipping():
    """Test that stress doesn't go negative at very high temperature."""
    params = JohnsonCookParams(
        A=500, B=300, n=0.15, C=0.0, m=1.0,
        epsdot0=1e-3, T_room=20, T_melt=1500
    )

    # At melting temperature, stress should be clipped to 0
    sigma = johnson_cook_flow_stress(0.10, 1e-3, 1500, params)[0]
    assert sigma >= 0


def test_negative_strain_rate_handling():
    """Test that negative strain rates are handled safely."""
    params = JohnsonCookParams(
        A=500, B=300, n=0.15, C=0.01, m=0.0,
        epsdot0=1e-3, T_room=20, T_melt=1500
    )

    # Should not crash, should use epsdot0 instead
    sigma = johnson_cook_flow_stress(0.10, -1, 20, params)[0]
    assert np.isfinite(sigma)
    assert sigma > 0


# ============================================================================
# Integration test - full workflow
# ============================================================================

def test_full_workflow():
    """Test complete workflow from standard to export (without actual file writing)."""
    # 1. Get spec
    spec = get_steel_spec("EC2", "B500C")

    # 2. Calibrate
    params = calibrate_jc_from_spec(spec, verbose=False)

    # 3. Generate curves
    results = generate_jc_curves_multicase(
        params=params,
        E=spec.E,
        strain_rates=[1e-3, 1],
        temperatures=[20, 400],
        n_points=50
    )

    # 4. Verify results structure
    assert len(results['curves']) == 4
    assert all('strain' in c for c in results['curves'])
    assert all('stress' in c for c in results['curves'])

    # 5. Check that stresses are reasonable
    for curve in results['curves']:
        assert np.all(curve['stress'] >= 0)
        assert np.all(curve['stress'] <= 2000)  # Reasonable max for steel

    print("✓ Full workflow test passed")


if __name__ == "__main__":
    # Run basic tests
    test_jc_flow_stress_monotonic()
    test_calibration_hits_fu()
    test_vectorization()
    test_full_workflow()
    print("\n✅ All basic tests passed!")
