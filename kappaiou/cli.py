"""

cli.py

backend input parsing

"""

import os
import sys
import argparse
import csv
from collections import defaultdict
from .core import calculate_adjusted_kappa_iou

def parse_csv_annotations(file_path):
    """
    Parses your standard input structure:
    Col 1: Filename | Col 2: Label (Object Type) | Col 3-6+: Spatial Data [x, y, w, h]

    Input:
    - file_path (str): path to annotations file

    Returns:
    - data (defaultdict): parsed annotations file
    """
    data = defaultdict(lambda: defaultdict(list))
  
    if not os.path.exists(file_path):
        print(f"Error: Target file not found at {file_path}")
        sys.exit(1)
        
    with open(file_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip optional headers if detected via string characters
        first_row = next(reader, None)
        
        if first_row:
            try:
                float(first_row[2])
                row_start = first_row
            except ValueError:
                row_start = None  # Row was a string header
                
            def _process_row(row):
                if len(row) < 6:
                    return
                
                filename = row[0]
                label = row[1]
              
                try:
                    coords = [float(c) for c in row[2:6]]
                    data[filename][label].append(coords)
                except ValueError:
                    pass # Row logging exception safety
                    
            if row_start:
                _process_row(row_start)
            for row in reader:
                _process_row(row)
              
    return data


def main():
    parser = argparse.ArgumentParser(description="Run Kappa-IoU verification against downstream annotator tranches.")
    parser.add_argument("--baseline",
                        required=True,
                        help="Path to your primary baseline CSV file.")
    parser.add_argument("--dir",
                        required=True,
                        help="Directory containing all other annotator CSV files.")
    parser.add_argument("--tau",
                        type=float,
                        default=0.70,
                        help="Expected spatial project acceptance threshold.")
    
    args = parser.parse_args()
    
    # Parse Master/Baseline Dataset Template
    baseline_data = parse_csv_annotations(args.baseline)
    
    # Gather other target CSV logs in target path directory
    all_files = os.listdir(args.dir)
    annotator_files = [f for f in all_files if f.endswith('.csv') and os.path.abspath(os.path.join(args.dir, f)) != os.path.abspath(args.baseline)]
    
    if not annotator_files:
        print(f"No valid secondary annotator CSV files located in target directory: {args.dir}")
        return

    print(f"=========================================================================")
    print(f"KAPPA-IoU COMPLIANCE LAB PIPELINE RUNTIME ENGINE")
    print(f"=========================================================================")
    print(f"Master baseline:       {os.path.basename(args.baseline)}")
    print(f"Operational tolerance (\u03c4): {args.tau}")
    print(f"Secondary systems found: {len(annotator_files)}")
    print(f"-------------------------------------------------------------------------")

    # Evaluate each annotator file independently
    for target in annotator_files:
        target_path = os.path.join(args.dir, target)
        annotator_data = parse_csv_annotations(target_path)
        
        cumulative_score = 0.0
        evaluation_channels = 0
        
        # Iterate cross-wise through every shared baseline domain image coordinate
        for filename, labels_dict in baseline_data.items():
            for label, base_boxes in labels_dict.items():
                # Extract corresponding logs from target annotator
                annotator_boxes = annotator_data.get(filename, {}).get(label, [])
                
                score = calculate_adjusted_kappa_iou(base_boxes, annotator_boxes, tau=args.tau)
                cumulative_score += score
                evaluation_channels += 1
                
        if evaluation_channels > 0:
            final_metric = cumulative_score / evaluation_channels
            print(f"Annotator File: {target:<30} | Adjusted Kappa-IoU: {final_metric:.4f}")
        else:
            print(f"Annotator File: {target:<30} | Adjusted Kappa-IoU: ERR (Zero matching image/label intersections)")

if __name__ == "__main__":
    main()
