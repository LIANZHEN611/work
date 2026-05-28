#!/usr/bin/env python3
"""
Fix KITTI calib files: recompute P2 matrix from intrinsic and extrinsic.
Usage:
    python fix_kitti_calib.py --calib_dir WARM-3D/data/s110_o_dataset/training/calib --intrinsic_json camera_params.json
"""

import argparse
import os
import glob
import json
import numpy as np

def parse_calib_line(line):
    """Parse line like 'P2: 1 2 3 ...' into numpy array."""
    parts = line.strip().split()
    name = parts[0].rstrip(':')
    values = list(map(float, parts[1:]))
    return name, np.array(values)

def load_intrinsic_from_json(json_path):
    """Load intrinsic camera matrix from JSON file (3x3)."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    K = np.array(data['intrinsic_camera_matrix'])  # 3x3
    return K

def compute_standard_p2(K, Tr_velo_to_cam):
    """
    Compute KITTI-style P2 matrix = K * [R|t] (3x4).
    Tr_velo_to_cam should be 3x4 or 4x4 (first 3 rows used).
    """
    if Tr_velo_to_cam.shape[0] == 4:
        Tr = Tr_velo_to_cam[:3, :]   # 3x4
    else:
        Tr = Tr_velo_to_cam          # assume 3x4
    P2 = K @ Tr   # 3x4
    return P2

def fix_calib_file(file_path, K, Tr_velo_to_cam, backup=True):
    """Fix a single calib file by replacing P2 line."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Backup original
    if backup:
        backup_path = file_path + '.bak'
        with open(backup_path, 'w') as f:
            f.writelines(lines)
        print(f"Backup saved to {backup_path}")

    # Find P2 line and replace
    new_lines = []
    p2_found = False
    for line in lines:
        if line.startswith('P2:'):
            new_P2 = compute_standard_p2(K, Tr_velo_to_cam)
            # Convert to list of strings with 12 decimal places
            p2_str = ' '.join(f"{x:.12f}" for x in new_P2.flatten())
            new_line = f"P2: {p2_str}\n"
            new_lines.append(new_line)
            p2_found = True
        else:
            new_lines.append(line)

    if not p2_found:
        print(f"Warning: No P2 line found in {file_path}, skipping.")
        return

    # Write back
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    print(f"Fixed {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Fix KITTI calib files P2 matrix')
    parser.add_argument('--calib_dir', required=True, help='Directory containing calib/*.txt files')
    parser.add_argument('--intrinsic_json', required=True, help='JSON file with intrinsic_camera_matrix')
    parser.add_argument('--no_backup', action='store_true', help='Do not create .bak backup')
    args = parser.parse_args()

    # Load intrinsic matrix
    K = load_intrinsic_from_json(args.intrinsic_json)
    print(f"Loaded intrinsic matrix:\n{K}")

    # Define the extrinsic matrix (Tr_velo_to_cam_2) for your camera
    # This is the transformation from LiDAR (north or south) to camera south2.
    # You should confirm this matches your setup.
    # Here we use the values from your conversion script for south2 camera.
    tr_velo_to_cam = np.array([
        [0.37383, -0.927155, 0.0251845, 14.2181],
        [-0.302544, -0.147564, -0.941643, 3.50648],
        [0.876766, 0.344395, -0.335669, -7.26891]
    ])   # 3x4
    print(f"Using extrinsic (Tr_velo_to_cam_2):\n{tr_velo_to_cam}")

    # Find all .txt files in calib directory
    calib_files = sorted(glob.glob(os.path.join(args.calib_dir, '*.txt')))
    if not calib_files:
        print(f"No .txt files found in {args.calib_dir}")
        return

    print(f"Found {len(calib_files)} calib files.")
    for f in calib_files:
        fix_calib_file(f, K, tr_velo_to_cam, backup=not args.no_backup)

    print("All done.")

if __name__ == '__main__':
    main()