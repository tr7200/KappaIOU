"""

This self-contained code recreates the results from section 4 of the paper:

Rogers, Trevor F. (2026). "Kappa-IoU: Inter-Rater Reliability for Spatial Annotation"


Images are drawn from the 2007 PascalVOC dataset: 

Everingham, M., Van Gool, L., Williams, C. K., Winn, J., & Zisserman, A. (2010). 
The pascal visual object classes (voc) challenge. 
International journal of computer vision, 88(2), 303-338.

"""



import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_iou(boxA, boxB):
    """
    Computes the Intersection-over-Union (IoU) between two bounding boxes.
    Boxes are expected in [x_min, y_min, width, height] format.

    Input:
    - boxA (list): list of xywh-format box annotations
    - boxB (list): list of xywh-format box annotations

    Returns:
    - Intersection-over-Union of boxes
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    
    union_area = boxA_area + boxB_area - intersection_area
    
    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area


def compute_image_kappa_iou(rater_a_boxes, 
                            rater_b_boxes, 
                            tau=0.70, 
                            delta=1e-7):
    """
    Executes the Hungarian assignment and computes the continuous spatial 
    observed agreement (P_0) and final Kappa-IoU for a single image.

    input:
    rater_a_boxes (list): list of lists of xyrb-format boxes, one list/object
    rater_b_boxes (list): same as above
    tau: desired agreement specified by project
    delta: stabilizing constant for Kappa-IoU denominator

    Returns:
    - P_0 agreement
    - Kappa-IoU metric
    - Assignments between annotations and objects
    """
    N_A = len(rater_a_boxes)
    N_B = len(rater_b_boxes)
    max_N = max(N_A, N_B)
    
    # Simplified handling of boundary case where raters agree there are zero objects (No adjusted Kappa-IoU)
    if max_N == 0:
        P_0 = 1.0 / (1.0 - 1.0 + delta)
        kappa = (P_0 - tau) / (1.0 - tau)
        
        return P_0, kappa, []

    # 1. Build dense asymmetric cost matrix (Dimensions: N_A x N_B)
    # Cost = 1 - IoU
    cost_matrix = np.zeros((N_A, N_B))
    for j in range(N_A):
        for k in range(N_B):
            cost_matrix[j, k] = 1.0 - calculate_iou(rater_a_boxes[j], rater_b_boxes[k])
            
    # 2. Rectangular-to-Square Normalization via Virtual Dummy Node Padding
    # Pad matrix to dimensions: max_N x max_N with static cost penalty of 1.0
    padded_cost = np.ones((max_N, max_N))
    padded_cost[:N_A, :N_B] = cost_matrix

    # 3. Execute Kuhn-Munkres Linear Assignment Optimization
    row_ind, col_ind = linear_sum_assignment(padded_cost)
    
    # Extract matched alignment paths and calculate optimized average IoU
    optimized_ious = []
    assignments = []
    
    for r, c in zip(row_ind, col_ind):
        # If assignment falls into the padded dummy zone, IoU is 0.0
        if r >= N_A or c >= N_B:
            iou = 0.0
            assignments.append((f"A_{r+1}" if r < N_A else "A_Ø", f"B_{c+1}" if c < N_B else "B_Ø", iou))
        else:
            iou = 1.0 - cost_matrix[r, c]
            assignments.append((f"A_{r+1}", f"B_{c+1}", iou))

        optimized_ious.append(iou)
        
    avg_iou = np.mean(optimized_ious)
    
    # 4. Compute Psychometric Metrics (P_0 and Chance-Corrected Kappa-IoU)
    P_0 = 1.0 / (1.0 - avg_iou + delta)
    kappa_IoU = (P_0 - tau) / (1.0 - tau)
    
    return P_0, kappa_IoU, assignments

# ==========================================
# EMPIRICAL VALIDATION: SECTION 4 TRANCHES
# ==========================================

# Project Configurations
TAU_TARGET = 0.70
DELTA_STABILIZER = 1e-7

# --- Tranche 1 Data (image of cat, 009528.jpg) ---
tranche1_rater_a = [[2.7439, 53.506, 373.6281, 273.018]]
tranche1_rater_b = [[75.457, 65.854, 418.903, 257.469]]

# --- Tranche 2 Data (image of two cows, 009908.jpg) ---
tranche2_rater_a = [
    [96.951, 105.64024, 156.403, 166.92076],  # Left Cow
    [246.494, 169.665, 143.14015, 105.18256]   # Right Cow
]
tranche2_rater_b = [
    [244.207, 173.323, 148.171, 101.982],      # Right Cow (Inverted Index)
    [94.207, 104.726, 160.061, 167.378]        # Left Cow (Inverted Index)
]

# --- Tranche 3 Data (image of three dogs, 009331.jpg) ---
tranche3_rater_a = [
    [92.835, 163.72, 126.22, 134.451],         # Left Running Dog
    [252.439, 194.36, 109.756, 80.488],        # Center Spotted Dog
    [325.61, 195.732, 91.463, 89.177]          # Right Spotted Dog
]
tranche3_rater_b = [
    [97.5, 168.0, 123.5, 125.0],               # Left Running Dog
    [322.5, 190.5, 93.0, 94.5],                # Right Spotted Dog (Inverted Index)
    [254.5, 196.5, 109.5, 83.5]                # Center Spotted Dog (Inverted Index)
]

# Run Pipeline Execution
datasets = {
    "Tranche 1: Single-Object Optimization (Cat)": (tranche1_rater_a, tranche1_rater_b),
    "Tranche 2: Multi-Object Invariant Sequencing (Cows)": (tranche2_rater_a, tranche2_rater_b),
    "Tranche 3: Symmetric Multi-Object Target Execution (Dogs)": (tranche3_rater_a, tranche3_rater_b)
}

print(f"=== Kappa-IoU Execution Engine (Target Threshold \u03c4 = {TAU_TARGET}) ===")
for title, (rater_a, rater_b) in datasets.items():
    P_0, kappa_IoU, paths = compute_image_kappa_iou(rater_a, rater_b, tau=TAU_TARGET, delta=DELTA_STABILIZER)
    
    print(f"\n\u25b6 {title}")
    print(f"  Calculated Observed Agreement (P_0): {P_0:.4f}")
    print(f"  Final Kappa-IoU Score (\u03ba_IoU):         {kappa_IoU:.4f}")
    print("  Optimal Linear Assignment Paths:")
    for src, dest, score in paths:
        print(f"    * {src} \u279f {dest} (Matched IoU = {score:.4f})")
