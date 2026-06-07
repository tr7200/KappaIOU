"""

Mathematical backend

- IoU
- Kappa-IoU
- adjusted Kappa-IoU (using modified BLEU for N-objects)

"""


import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_iou(boxA, boxB):
    """
    Computes Intersection-over-Union (IoU) between two boxes in [xmin, ymin, w, h] format.

    Input:
    - boxA (list): xywh-format annotation
    - boxB (list): xywh-format annotation

    Returns:
    - Intersection-over-Union of boxA and boxB
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
    In case of raters annotating different numbers of objects on an image, 
    this difference is solved as a linear assignment problem via Kuhn-Munkres 
    and Kappa-IoU is calculated for a single object-set channel.
    
    Input:
    - rater_a_boxes (List): List of lists of boxes annotated by rater A
    - rater_b_boxes (List): List of lists of boxes annotated by rater B
    - tau (float): P_T, the acceptable threshold of agreement between rater A and rater B
    - delta (float): stabilizing constant to prevent division by zero

    Returns:
    - Kappa-IoU metric of rater A and rater B
    """
    N_A = len(rater_a_boxes)
    N_B = len(rater_b_boxes)
    max_N = max(N_A, N_B)
    
    if max_N == 0:
        return 1.0  # Perfect consensus results on an empty image channel
        
    # Build cost matrix (1 - IoU)
    cost_matrix = np.zeros((N_A, N_B))
    for j in range(N_A):
        for k in range(N_B):
            cost_matrix[j, k] = 1.0 - calculate_iou(rater_a_boxes[j], rater_b_boxes[k])
            
    # Pad to square configuration with max cost penalty of 1.0
    padded_cost = np.ones((max_N, max_N))
    padded_cost[:N_A, :N_B] = cost_matrix
    
    # Primal-dual optimization path routing
    row_ind, col_ind = linear_sum_assignment(padded_cost)
    
    optimized_ious = []
    for r, c in zip(row_ind, col_ind):
        if r >= N_A or c >= N_B:
            optimized_ious.append(0.0)  # Unmatched dummy penalty
        else:
            optimized_ious.append(1.0 - cost_matrix[r, c])
            
    avg_iou = np.mean(optimized_ious)
                              
    P_0 = 1.0 / (1.0 - avg_iou + delta)
    kappaiou = (P_0 - tau) / (1.0 - tau)
                              
    return kappaiou


def calculate_adjusted_kappa_iou(rater_a_boxes, 
                                 rater_b_boxes, 
                                 tau=0.70, 
                                 delta=1e-7):
    """
    Implements the BLEU-adapted geometric ranking summation framework.
    Evaluates sub-sequences from 1 to N-objects and aggregates the reliability index.
    
    Input:
    - rater_a_boxes (List): List of lists of boxes annotated by rater A
    - rater_b_boxes (List): List of lists of boxes annotated by rater B
    - tau (float): P_T, the acceptable threshold of agreement between rater A and rater B
    - delta (float): stabilizing constant to prevent division by zero

    Returns:
    - Kappa-IoU metric of rater A and rater B adjusted by modified BLEU
    """
    N_A = len(rater_a_boxes)
    N_B = len(rater_b_boxes)
    N = max(N_A, N_B)
    
    if N == 0:
        return 0.0
        
    kappa_terms = []
                                   
    # Evaluate sequentially across every depth layer from 1 to N
    for n in range(1, N + 1):
        sub_a = rater_a_boxes[:n]
        sub_b = rater_b_boxes[:n]
        k_n = compute_image_kappa_iou(sub_a, sub_b, tau=tau, delta=delta)
        kappa_terms.append(k_n)
        
    # Structural geometric log-precision summation sequence
    # Protects against taking natural logs of zero or negative Kappa outputs
    valid_logs = [np.log(k) if k > 0 else np.log(delta) for k in kappa_terms]
    adjusted_score = np.exp(np.sum((1.0 / N) * np.array(valid_logs)))
                                   
    return adjusted_score
