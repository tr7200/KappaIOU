"""

Unit tests

"""

import pytest
import numpy as np

from kappaiou.core import (
    calculate_iou, 
    compute_image_kappa_iou, 
    calculate_adjusted_kappa_iou
)

# =========================================================================
# EMPIRICAL TRUNCATED RAW DATA CONSTANTS (FROM MANUSCRIPT SECTION 4)
# =========================================================================

# Tranche 1: Single-Object (Cat)
T1_RATER_A = [[2.7439, 53.506, 373.6281, 273.018]]
T1_RATER_B = [[75.457, 65.854, 418.903, 257.469]]

# Tranche 2: Multi-Object Inverted Sequence (Cows)
T2_RATER_A = [
    [96.951, 105.64024, 156.403, 166.92076],  # Left Cow
    [246.494, 169.665, 143.14015, 105.18256]   # Right Cow
]
T2_RATER_B = [
    [244.207, 173.323, 148.171, 101.982],      # Right Cow (Inverted)
    [94.207, 104.726, 160.061, 167.378]        # Left Cow (Inverted)
]

# Tranche 3: Multi-Object Missingness/Symmetric Execution (Dogs)
T3_RATER_A = [
    [92.835, 163.72, 126.22, 134.451],         # Left Running Dog
    [252.439, 194.36, 109.756, 80.488],        # Center Spotted Dog
    [325.61, 195.732, 91.463, 89.177]          # Right Spotted Dog
]
T3_RATER_B = [
    [97.5, 168.0, 123.5, 125.0],               # Left Running Dog
    [322.5, 190.5, 93.0, 94.5],                # Right Spotted Dog (Inverted)
    [254.5, 196.5, 109.5, 83.5]                # Center Spotted Dog (Inverted)
]

# =========================================================================
# PYTEST COMPLIANCE LAB TEST SUITE
# =========================================================================

def test_tranche_1_single_object_scaling():
    """
    Verifies that Tranche 1 (Cat) properly exhibits the open-ended positive
    scaling characteristics when P_0 exceeds classical limits.
    """
    tau = 0.70
    delta = 1e-7
    
    # Extract structural baseline behavior
    kappa_score = compute_image_kappa_iou(T1_RATER_A, T1_RATER_B, tau=tau, delta=delta)
    
    # Assertions grounded in Section 4 empirical findings
    assert kappa_score > 1.0, f"Expected Kappa to break upper bound, got {kappa_score}"
    assert np.isfinite(kappa_score), "Kappa calculation returned unstable non-finite float."


def test_tranche_2_sequence_invariance():
    """
    Verifies that the Kuhn-Munkres engine unscrambles arbitrary sequence inversions,
    yielding a highly stable score regardless of rater entry order.
    """
    tau = 0.70
    delta = 1e-7
    
    # Run structural assessment on the cows (inverted indexes)
    kappa_score = compute_image_kappa_iou(T2_RATER_A, T2_RATER_B, tau=tau, delta=delta)
    
    # Construct a perfectly sequential reference to verify invariance
    sequenced_rater_b = [T2_RATER_B[1], T2_RATER_B[0]]
    reference_score = compute_image_kappa_iou(T2_RATER_A, sequenced_rater_b, tau=tau, delta=delta)
    
    # Assert that the optimization resolves order variance completely
    assert kappa_score == pytest.approx(reference_score, rel=1e-6)


def test_tranche_3_adjusted_sequential_bleu():
    """
    Verifies that the multi-object BLEU aggregation accurately combines 
    sub-sequence Kappa scores across varying depths up to N objects.
    """
    tau = 0.70
    delta = 1e-7
    
    # Run full adjusted pipeline on the dogs dataset
    adjusted_kappa = calculate_adjusted_kappa_iou(T3_RATER_A, T3_RATER_B, tau=tau, delta=delta)
    
    # Verify that the nested geometric logging yields a stable bounded result
    assert isinstance(adjusted_kappa, float)
    assert adjusted_kappa > 0.0, f"Expected valid positive adjusted matrix score, got {adjusted_kappa}"
