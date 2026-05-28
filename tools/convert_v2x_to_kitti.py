import argparse
import json
import os
import shutil
import sys

import numpy as np


_K_S1_R04 = np.array([[-1301.42, 0, 940.389], [0, -1299.94, 674.417], [0, 0, 1]], dtype=np.float64)
_E_S1_R04 = np.array([
    [-0.41205, 0.910783, -0.0262516, 15.0787],
    [0.453777, 0.230108, 0.860893, 2.52926],
    [0.790127, 0.342818, -0.508109, 3.67868],
], dtype=np.float64)

_K_NORTH = np.array([[1360.68, 0, 849.369], [0, 1470.71, 632.174], [0, 0, 1]], dtype=np.float64)
_E_NORTH = np.array([
    [-0.564602, -0.824833, -0.0295815, -12.9358],
    [-0.458346, 0.343143, -0.819861, 7.22666],
    [0.686399, -0.449337, -0.571798, -6.75018],
], dtype=np.float64)

PROJECTION_MATRICES = {
    's110_camera_basler_south1_8mm': _K_S1_R04 @ _E_S1_R04,
    's110_camera_basler_south2_8mm': np.array([
        [1318.95273325, -859.15213894, -289.13390611, 11272.03223502],
        [90.01799314, -2.9727517, -1445.63809767, 585.78988153],
        [0.876766, 0.344395, -0.335669, -7.26891],
    ], dtype=np.float64),
    's110_camera_basler_east_8mm': np.array([
        [-2666.70160799, -655.44528859, -790.96345758, -33010.77350141],
        [430.89231274, 66.06703744, -2053.70223986, 6630.65222157],
        [-0.00932524, -0.96164431, -0.27414094, 11.41820108],
    ], dtype=np.float64),
    's110_camera_basler_north_8mm': _K_NORTH @ _E_NORTH,
}

CAMERA_TO_OUT_DIR = {
    's110_camera_basler_north_8mm': 's110_n_dataset',
    's110_camera_basler_east_8mm':  's110_o_dataset',
    's110_camera_basler_south1_8mm': 's110_s_dataset',
    's110_camera_basler_south2_8mm': 's110_s2_dataset',
}

OCCLUSION_MAP = {'NOT_OCCLUDED': 0, 'PARTIALLY_OCCLUDED': 1, 'MOSTLY_OCCLUDED': 2}

CLASS_NAME_MAP = {
    'CAR': 'Car',
    'VAN': 'Van',
    'BUS': 'Bus',
    'TRUCK': 'BigCar',
    'TRAILER': 'BigCar',
    'BIGCAR': 'BigCar',
    'PEDESTRIAN': 'Pedestrian',
    'CYCLIST': 'Cyclist',
    'MOTORCYCLE': 'Cyclist',
    'BICYCLE': 'Cyclist',
}


def quat_to_rot(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def yaw_from_quat(qx, qy, qz, qw):
    sin_y = 2.0 * (qw * qz + qx * qy)
    cos_y = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(sin_y, cos_y))


def project_cuboid_to_2d(cuboid_val, P, img_w=1920, img_h=1200):
    x, y, z = cuboid_val[0], cuboid_val[1], cuboid_val[2]
    qx, qy, qz, qw = cuboid_val[3:7]
    l, w, h = cuboid_val[7], cuboid_val[8], cuboid_val[9]
    R = quat_to_rot(qx, qy, qz, qw)
    corners_obj = np.array([
        [ l/2,  w/2,  h/2], [ l/2,  w/2, -h/2], [ l/2, -w/2,  h/2], [ l/2, -w/2, -h/2],
        [-l/2,  w/2,  h/2], [-l/2,  w/2, -h/2], [-l/2, -w/2,  h/2], [-l/2, -w/2, -h/2],
    ], dtype=np.float64)
    corners_lidar = (R @ corners_obj.T).T + np.array([x, y, z])
    ones = np.ones((8, 1))
    proj = (P @ np.hstack([corners_lidar, ones]).T).T
    valid = proj[:, 2] > 1e-3
    if not np.any(valid):
        return None
    proj = proj[valid]
    px = proj[:, 0] / proj[:, 2]
    py = proj[:, 1] / proj[:, 2]
    xmin = float(np.clip(np.min(px), 0, img_w - 1))
    ymin = float(np.clip(np.min(py), 0, img_h - 1))
    xmax = float(np.clip(np.max(px), 0, img_w - 1))
    ymax = float(np.clip(np.max(py), 0, img_h - 1))
    if xmax <= xmin + 1 or ymax <= ymin + 1:
        return None
    return [xmin, ymin, xmax, ymax]


def write_calib_kitti(out_path, P):
    Pflat = ' '.join(f'{v:.8f}' for v in P.reshape(-1))
    eye_flat = '1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0'
    tr_flat = '0.0 -1.0 0.0 0.0 0.0 0.0 -1.0 -0.0 1.0 0.0 0.0 0.0'
    imu_flat = '1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0'
    with open(out_path, 'w') as f:
        for tag in ('P0', 'P1', 'P2', 'P3'):
            f.write(f'{tag}: {Pflat}\n')
        f.write(f'R0_rect: {eye_flat}\n')
        f.write(f'Tr_velo_to_cam: {tr_flat}\n')
        f.write(f'TR_imu_to_velo: {imu_flat}\n')


def write_extrinsics_stub(out_path):
    stub = {
        'base_to_camera_matrix': np.eye(4).tolist(),
    }
    with open(out_path, 'w') as f:
        json.dump(stub, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v2x-root', default='data/tumtraf_v2x_cooperative')
    p.add_argument('--out-root', default='data/v2x_kitti')
    p.add_argument('--split', default='val')
    p.add_argument('--img-w', type=int, default=1920)
    p.add_argument('--img-h', type=int, default=1200)
    args = p.parse_args()

    label_dir = os.path.join(args.v2x_root, args.split, 'labels_point_clouds',
                             's110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
    label_files = sorted(f for f in os.listdir(label_dir) if f.endswith('.json'))
    print(f'Found {len(label_files)} label files in {label_dir}')

    for cam in CAMERA_TO_OUT_DIR:
        ds = os.path.join(args.out_root, CAMERA_TO_OUT_DIR[cam])
        os.makedirs(os.path.join(ds, 'training', 'image_2'), exist_ok=True)
        os.makedirs(os.path.join(ds, 'training', 'label_2'), exist_ok=True)
        os.makedirs(os.path.join(ds, 'training', 'calib'), exist_ok=True)
        os.makedirs(os.path.join(ds, 'training', 'extrinsics'), exist_ok=True)
        os.makedirs(os.path.join(ds, 'ImageSets'), exist_ok=True)

    img_root = os.path.join(args.v2x_root, args.split, 'images')

    per_cam_index = {c: 0 for c in CAMERA_TO_OUT_DIR}
    per_cam_imageset = {c: [] for c in CAMERA_TO_OUT_DIR}
    n_total_objs = 0
    n_proj_ok = 0

    for label_fn in label_files:
        with open(os.path.join(label_dir, label_fn)) as f:
            data = json.load(f)
        frames = data['openlabel']['frames']
        frame_key = next(iter(frames))
        frame = frames[frame_key]
        image_names = frame['frame_properties'].get('image_file_names', [])
        objs = frame.get('objects', {}) or {}

        camera_to_img_name = {}
        for img_name in image_names:
            for cam in CAMERA_TO_OUT_DIR:
                if cam in img_name:
                    camera_to_img_name[cam] = img_name
                    break

        for cam in CAMERA_TO_OUT_DIR:
            if cam not in camera_to_img_name:
                continue
            img_name = camera_to_img_name[cam]
            src_img = os.path.join(img_root, cam, img_name)
            if not os.path.isfile(src_img):
                continue
            P = PROJECTION_MATRICES[cam]
            idx = per_cam_index[cam]
            stem = f'{idx:06d}'
            ds = os.path.join(args.out_root, CAMERA_TO_OUT_DIR[cam], 'training')

            shutil.copy(src_img, os.path.join(ds, 'image_2', f'{stem}.jpg'))
            write_calib_kitti(os.path.join(ds, 'calib', f'{stem}.txt'), P)
            write_extrinsics_stub(os.path.join(ds, 'extrinsics', f'{stem}.json'))

            lines = []
            for obj in objs.values():
                od = obj['object_data']
                raw_category = od['type']
                category = CLASS_NAME_MAP.get(raw_category.upper(), None)
                if category is None:
                    continue
                cb = od['cuboid']
                if isinstance(cb, list):
                    cb = cb[0]
                cuboid_val = cb['val']
                occluded = 0
                for item in cb.get('attributes', {}).get('text', []):
                    if item['name'] == 'occlusion_level':
                        occluded = OCCLUSION_MAP.get(item['val'], 0)
                bbox2d = project_cuboid_to_2d(cuboid_val, P, args.img_w, args.img_h)
                n_total_objs += 1
                if bbox2d is None:
                    continue
                n_proj_ok += 1
                x_c, y_c, z_c = cuboid_val[0], cuboid_val[1], cuboid_val[2]
                l, w, h = cuboid_val[7], cuboid_val[8], cuboid_val[9]
                yaw = yaw_from_quat(cuboid_val[3], cuboid_val[4], cuboid_val[5], cuboid_val[6])
                truncated = 0.0
                alpha = 0.0
                line = (
                    f'{category} {truncated:.2f} {occluded} {alpha:.2f} '
                    f'{bbox2d[0]:.2f} {bbox2d[1]:.2f} {bbox2d[2]:.2f} {bbox2d[3]:.2f} '
                    f'{h:.2f} {w:.2f} {l:.2f} {x_c:.2f} {y_c:.2f} {z_c:.2f} {yaw:.2f} 0.0 0.0\n'
                )
                lines.append(line)
            with open(os.path.join(ds, 'label_2', f'{stem}.txt'), 'w') as f:
                f.writelines(lines)

            per_cam_imageset[cam].append(stem)
            per_cam_index[cam] += 1

    for cam, ids in per_cam_imageset.items():
        ds = os.path.join(args.out_root, CAMERA_TO_OUT_DIR[cam])
        for name in ('train', 'val', 'trainval'):
            with open(os.path.join(ds, 'ImageSets', f'{name}.txt'), 'w') as f:
                f.write('\n'.join(ids) + '\n')
        print(f'  {cam:<40s} -> {ds}/ ({len(ids)} frames)')

    print(f'projected ok / total: {n_proj_ok}/{n_total_objs}')


if __name__ == '__main__':
    main()
