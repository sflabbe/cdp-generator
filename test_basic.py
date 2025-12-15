"""
Basic test to verify package functionality
"""

from cdp_generator import (
    calculate_stress_strain,
    calculate_concrete_strength_properties,
    calculate_elastic_modulus
)


def test_basic_calculation():
    """Test basic calculation without plotting or Excel export."""
    print("Testing basic CDP calculation...")

    # Input parameters
    f_cm = 28.0
    e_c1 = 0.0022
    e_clim = 0.0035
    l_ch = 1.0
    strain_rates = [0]  # Only static case

    # Calculate stress-strain relationships
    results = calculate_stress_strain(f_cm, e_c1, e_clim, l_ch, strain_rates)

    # Verify results structure
    assert 'properties' in results
    assert 'compression' in results
    assert 'tension' in results

    # Verify properties
    props = results['properties']
    print(f"\n✓ Material Properties:")
    print(f"  Tensile strength: {props['tensile strength']:.2f} MPa")
    print(f"  Elastic modulus: {props['elasticity']:.2f} MPa")
    print(f"  Dilation angle: {props['dilation angle']:.2f}°")
    print(f"  CDP Kc: {props['Kc']:.2f}")
    print(f"  CDP fb/fc: {props['fbfc']:.2f}")

    # Basic sanity checks
    assert props['tensile strength'] > 0
    assert props['elasticity'] > 0
    assert props['dilation angle'] > 0

    print("\n✓ All basic tests passed!")


def test_individual_functions():
    """Test individual module functions."""
    print("\nTesting individual functions...")

    f_cm = 28.0

    # Test strength properties
    strength_props = calculate_concrete_strength_properties(f_cm)
    assert 'f_ck' in strength_props
    assert 'f_ctm' in strength_props
    print(f"✓ Strength properties: f_ck={strength_props['f_ck']:.2f} MPa, f_ctm={strength_props['f_ctm']:.2f} MPa")

    # Test elastic modulus
    elastic_props = calculate_elastic_modulus(f_cm)
    assert 'E_ci' in elastic_props
    assert 'E_c' in elastic_props
    print(f"✓ Elastic modulus: E_ci={elastic_props['E_ci']:.2f} MPa, E_c={elastic_props['E_c']:.2f} MPa")

    print("✓ Individual function tests passed!")


if __name__ == "__main__":
    test_basic_calculation()
    test_individual_functions()
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)
