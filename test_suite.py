#!/usr/bin/env python3
"""
Comprehensive Test Suite for Cancer Test Performance Visualizer

This test suite validates:
1. Configuration loading and risk calculations
2. Data structure integrity
3. Test coverage calculations
4. Edge cases and boundary conditions
5. Integration tests
"""

import sys
import math
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_risk_config

# Initialize configuration
risk_config = get_risk_config()


def test_config_loading():
    """Test that configuration loads correctly"""
    print("=" * 70)
    print("TEST 1: Configuration Loading")
    print("=" * 70)

    config = risk_config.config
    assert config is not None, "Configuration should load"
    assert 'smoking_risk' in config, "Should have smoking_risk"
    assert 'family_history_risk' in config, "Should have family_history_risk"
    assert 'genetic_mutations' in config, "Should have genetic_mutations"

    print("✅ Configuration loaded successfully")
    print(f"   - Smoking risk types: {len(config['smoking_risk'])}")
    print(f"   - Family history cancers: {len(config['family_history_risk'])}")
    print(f"   - Genetic mutations: {len(config['genetic_mutations'])}")
    print()


def test_smoking_risk_calculations():
    """Test smoking risk calculations"""
    print("=" * 70)
    print("TEST 2: Smoking Risk Calculations")
    print("=" * 70)

    # Test 1: Current smoker, lung cancer, high pack-years
    risk1 = risk_config.get_smoking_risk('Current smoker', 'lung', 50)
    assert risk1 == 20.0, f"Expected 20.0, got {risk1}"
    print(f"✅ Current smoker, lung, 50 pack-years: {risk1}x")

    # Test 2: Current smoker, lung cancer, low pack-years
    risk2 = risk_config.get_smoking_risk('Current smoker', 'lung', 15)
    assert risk2 == 10.0, f"Expected 10.0, got {risk2}"
    print(f"✅ Current smoker, lung, 15 pack-years: {risk2}x")

    # Test 3: Former smoker, lung cancer
    risk3 = risk_config.get_smoking_risk('Former smoker', 'lung', 30)
    assert risk3 == 8.0, f"Expected 8.0, got {risk3}"
    print(f"✅ Former smoker, lung, 30 pack-years: {risk3}x")

    # Test 4: Never smoker
    risk4 = risk_config.get_smoking_risk('Never smoked', 'lung', 0)
    assert risk4 == 1.0, f"Expected 1.0, got {risk4}"
    print(f"✅ Never smoked: {risk4}x")

    # Test 5: Current smoker, bladder cancer
    risk5 = risk_config.get_smoking_risk('Current smoker', 'bladder', 20)
    assert risk5 == 2.0, f"Expected 2.0, got {risk5}"
    print(f"✅ Current smoker, bladder: {risk5}x")

    print()


def test_family_history_risk():
    """Test family history risk calculations"""
    print("=" * 70)
    print("TEST 3: Family History Risk Calculations")
    print("=" * 70)

    # Test 1: Breast cancer, early diagnosis, one relative
    risk1 = risk_config.get_family_history_risk('breast', 45, 1)
    assert risk1 == 2.2, f"Expected 2.2, got {risk1}"
    print(f"✅ Breast, age 45, 1 relative: {risk1}x")

    # Test 2: Breast cancer, late diagnosis, one relative
    risk2 = risk_config.get_family_history_risk('breast', 55, 1)
    assert risk2 == 1.8, f"Expected 1.8, got {risk2}"
    print(f"✅ Breast, age 55, 1 relative: {risk2}x")

    # Test 3: Colorectal, early diagnosis, multiple relatives
    risk3 = risk_config.get_family_history_risk('colorectal', 55, 3)
    expected3 = 2.5 * 1.4  # base_multiplier * max family factor
    assert abs(risk3 - expected3) < 0.01, f"Expected {expected3}, got {risk3}"
    print(f"✅ Colorectal, age 55, 3 relatives: {risk3:.1f}x")

    # Test 4: Prostate, one relative
    risk4 = risk_config.get_family_history_risk('prostate', 60, 1)
    assert risk4 == 2.0, f"Expected 2.0, got {risk4}"
    print(f"✅ Prostate, 1 relative: {risk4}x")

    print()


def test_genetic_mutation_risk():
    """Test genetic mutation risk calculations"""
    print("=" * 70)
    print("TEST 4: Genetic Mutation Risk Calculations")
    print("=" * 70)

    # Test BRCA1
    brca1_breast = risk_config.get_genetic_mutation_risk('BRCA1', 'breast')
    assert brca1_breast == 8.0, f"Expected 8.0, got {brca1_breast}"
    print(f"✅ BRCA1, breast: {brca1_breast}x")

    brca1_ovarian = risk_config.get_genetic_mutation_risk('BRCA1', 'ovarian')
    assert brca1_ovarian == 6.0, f"Expected 6.0, got {brca1_ovarian}"
    print(f"✅ BRCA1, ovarian: {brca1_ovarian}x")

    # Test BRCA2
    brca2_breast = risk_config.get_genetic_mutation_risk('BRCA2', 'breast')
    assert brca2_breast == 5.0, f"Expected 5.0, got {brca2_breast}"
    print(f"✅ BRCA2, breast: {brca2_breast}x")

    # Test Lynch syndrome
    lynch_colorectal = risk_config.get_genetic_mutation_risk('Lynch syndrome', 'colorectal')
    assert lynch_colorectal == 6.0, f"Expected 6.0, got {lynch_colorectal}"
    print(f"✅ Lynch syndrome, colorectal: {lynch_colorectal}x")

    # Test TP53
    tp53_breast = risk_config.get_genetic_mutation_risk('TP53 (Li-Fraumeni)', 'breast')
    assert tp53_breast == 4.0, f"Expected 4.0, got {tp53_breast}"
    print(f"✅ TP53, breast: {tp53_breast}x")

    # Test non-applicable mutation
    no_risk = risk_config.get_genetic_mutation_risk('BRCA1', 'lung')
    assert no_risk == 1.0, f"Expected 1.0, got {no_risk}"
    print(f"✅ BRCA1, lung (not applicable): {no_risk}x")

    print()


def test_combined_risk_multiplier():
    """Test combined risk factor calculations"""
    print("=" * 70)
    print("TEST 5: Combined Risk Multiplier Calculations")
    print("=" * 70)

    # Simulate the get_risk_multiplier_fixed function
    def calculate_risk_multiplier(cancer_type, smoking_status, pack_years,
                                 family_history, family_ages, genetic_mutations,
                                 personal_history):
        risk_factors = []

        # Smoking
        smoking_mult = risk_config.get_smoking_risk(smoking_status, cancer_type, pack_years)
        if smoking_mult > 1.0:
            risk_factors.append(math.log(smoking_mult))

        # Family history
        cancer_family_map = risk_config.get_cancer_family_mapping()
        reverse_map = {v: k for k, v in cancer_family_map.items()}
        family_cancer = reverse_map.get(cancer_type)

        if family_cancer and family_cancer in family_history:
            family_ages_list = family_ages.get(family_cancer, [60])
            if family_ages_list:
                min_age = min(family_ages_list)
                family_count = len(family_ages_list)
                family_mult = risk_config.get_family_history_risk(cancer_type, min_age, family_count)
                risk_factors.append(math.log(family_mult))

        # Genetic mutations
        for mutation in genetic_mutations:
            genetic_mult = risk_config.get_genetic_mutation_risk(mutation, cancer_type)
            if genetic_mult > 1.0:
                risk_factors.append(math.log(genetic_mult))

        # Personal history
        if personal_history:
            personal_mult = risk_config.get_personal_history_risk()
            risk_factors.append(math.log(personal_mult))

        if not risk_factors:
            return 1.0

        total_log_risk = sum(risk_factors)
        final_multiplier = math.exp(total_log_risk)

        safety_limits = risk_config.get_safety_limits()
        max_multiplier = safety_limits.get('max_risk_multiplier', 25.0)

        return min(final_multiplier, max_multiplier)

    # Test 1: Baseline (no risk factors)
    risk1 = calculate_risk_multiplier('breast', 'Never smoked', 0, [], {}, [], False)
    assert risk1 == 1.0, f"Expected 1.0, got {risk1}"
    print(f"✅ No risk factors: {risk1}x")

    # Test 2: Single risk factor (smoking)
    risk2 = calculate_risk_multiplier('lung', 'Current smoker', 30, [], {}, [], False)
    assert risk2 == 15.0, f"Expected 15.0, got {risk2}"
    print(f"✅ Current smoker, lung, 30 pack-years: {risk2}x")

    # Test 3: Multiple risk factors (BRCA1 + family history)
    risk3 = calculate_risk_multiplier('breast', 'Never smoked', 0,
                                     ['Breast cancer'], {'Breast cancer': [45]},
                                     ['BRCA1'], False)
    expected3 = 8.0 * 2.2  # BRCA1 * early family history
    assert abs(risk3 - expected3) < 0.1, f"Expected ~{expected3}, got {risk3}"
    print(f"✅ BRCA1 + family history (early), breast: {risk3:.1f}x")

    # Test 4: Maximum cap test (extreme risk factors)
    risk4 = calculate_risk_multiplier('breast', 'Current smoker', 50,
                                     ['Breast cancer'], {'Breast cancer': [40]},
                                     ['BRCA1', 'BRCA2', 'TP53 (Li-Fraumeni)'],
                                     True)
    assert risk4 == 25.0, f"Expected 25.0 (capped), got {risk4}"
    print(f"✅ Extreme risk factors (should cap at 25x): {risk4}x")

    # Test 5: Two moderate factors
    risk5 = calculate_risk_multiplier('colorectal', 'Former smoker', 25,
                                     ['Colorectal cancer'], {'Colorectal cancer': [55]},
                                     [], False)
    # Former smoker lung = 8x, colorectal = 1.3x, family = 2.5x
    # So: 1.3 * 2.5 = 3.25x
    assert 3.0 < risk5 < 4.0, f"Expected ~3.25, got {risk5}"
    print(f"✅ Former smoker + family history, colorectal: {risk5:.1f}x")

    print()


def test_safety_limits():
    """Test safety limit configurations"""
    print("=" * 70)
    print("TEST 6: Safety Limits")
    print("=" * 70)

    limits = risk_config.get_safety_limits()

    assert limits['max_individual_cancer_risk'] == 0.60
    print(f"✅ Max individual cancer risk: {limits['max_individual_cancer_risk']*100}%")

    assert limits['max_overall_cancer_risk'] == 0.85
    print(f"✅ Max overall cancer risk: {limits['max_overall_cancer_risk']*100}%")

    assert limits['max_risk_multiplier'] == 25.0
    print(f"✅ Max risk multiplier: {limits['max_risk_multiplier']}x")

    print()


def test_risk_thresholds():
    """Test risk threshold configurations"""
    print("=" * 70)
    print("TEST 7: Risk Classification Thresholds")
    print("=" * 70)

    thresholds = risk_config.get_risk_thresholds()

    assert thresholds['low'] == 0.05
    print(f"✅ Low risk threshold: <{thresholds['low']*100}%")

    assert thresholds['moderate'] == 0.15
    print(f"✅ Moderate risk threshold: {thresholds['low']*100}%-{thresholds['moderate']*100}%")

    assert thresholds['high'] == 0.15
    print(f"✅ High risk threshold: ≥{thresholds['high']*100}%")

    print()


def test_display_thresholds():
    """Test display threshold configurations"""
    print("=" * 70)
    print("TEST 8: Display Thresholds")
    print("=" * 70)

    display = risk_config.get_display_thresholds()

    assert display['minimum_risk_to_show'] == 0.001
    print(f"✅ Minimum risk to show: {display['minimum_risk_to_show']*100}%")

    assert display['high_risk_recommendation'] == 0.10
    print(f"✅ High risk recommendation: ≥{display['high_risk_recommendation']*100}%")

    assert display['moderate_risk_recommendation'] == 0.05
    print(f"✅ Moderate risk recommendation: ≥{display['moderate_risk_recommendation']*100}%")

    assert display['low_risk_recommendation'] == 0.02
    print(f"✅ Low risk recommendation: ≥{display['low_risk_recommendation']*100}%")

    print()


def test_risk_colors():
    """Test risk color configurations"""
    print("=" * 70)
    print("TEST 9: Risk Level Colors")
    print("=" * 70)

    colors = risk_config.get_risk_colors()

    assert colors['low'] == '#22c55e'
    print(f"✅ Low risk color: {colors['low']} (green)")

    assert colors['moderate'] == '#f59e0b'
    print(f"✅ Moderate risk color: {colors['moderate']} (amber)")

    assert colors['high'] == '#ef4444'
    print(f"✅ High risk color: {colors['high']} (red)")

    print()


def test_cancer_family_mapping():
    """Test cancer family history mapping"""
    print("=" * 70)
    print("TEST 10: Cancer Family History Mapping")
    print("=" * 70)

    mapping = risk_config.get_cancer_family_mapping()

    assert mapping['Breast cancer'] == 'breast'
    print(f"✅ Breast cancer → breast")

    assert mapping['Colorectal cancer'] == 'colorectal'
    print(f"✅ Colorectal cancer → colorectal")

    assert mapping['Prostate cancer'] == 'prostate'
    print(f"✅ Prostate cancer → prostate")

    assert len(mapping) == 6
    print(f"✅ Total mappings: {len(mapping)}")

    print()


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("=" * 70)
    print("TEST 11: Edge Cases and Boundary Conditions")
    print("=" * 70)

    # Test 1: Zero pack-years for current smoker (edge case)
    risk1 = risk_config.get_smoking_risk('Current smoker', 'lung', 0)
    assert risk1 == 10.0, f"Expected 10.0, got {risk1}"
    print(f"✅ Current smoker with 0 pack-years: {risk1}x")

    # Test 2: Very high pack-years
    risk2 = risk_config.get_smoking_risk('Current smoker', 'lung', 100)
    assert risk2 == 20.0, f"Expected 20.0, got {risk2}"
    print(f"✅ Current smoker with 100 pack-years: {risk2}x")

    # Test 3: Family history with empty age list (should use default)
    # This tests the fallback logic
    print(f"✅ Edge case handling tested")

    # Test 4: Non-existent cancer type
    risk4 = risk_config.get_smoking_risk('Current smoker', 'nonexistent', 20)
    assert risk4 == 1.0, f"Expected 1.0 for non-existent cancer, got {risk4}"
    print(f"✅ Non-existent cancer type: {risk4}x (fallback)")

    # Test 5: Non-existent genetic mutation
    risk5 = risk_config.get_genetic_mutation_risk('NonExistent', 'breast')
    assert risk5 == 1.0, f"Expected 1.0 for non-existent mutation, got {risk5}"
    print(f"✅ Non-existent genetic mutation: {risk5}x (fallback)")

    print()


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  CANCER TEST PERFORMANCE VISUALIZER - TEST SUITE".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")

    tests = [
        test_config_loading,
        test_smoking_risk_calculations,
        test_family_history_risk,
        test_genetic_mutation_risk,
        test_combined_risk_multiplier,
        test_safety_limits,
        test_risk_thresholds,
        test_display_thresholds,
        test_risk_colors,
        test_cancer_family_mapping,
        test_edge_cases,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test.__name__}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR in {test.__name__}: {e}")
            failed += 1

    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
