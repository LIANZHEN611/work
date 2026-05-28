from glob import glob

import cv2
#from pypcd import pypcd
from tqdm import tqdm
import argparse
import json
import ntpath
import numpy as np
import os.path
import shutil


# Module description:
# This module converts the OpenLABEL format to KITTI format.

# Requirements:
#   - The dataset needs to be in the OpenLABEL format.
#   - The dataset needs to be split into a training and validation set: src/preprocessing/create_train_val_split.py

# Usage:
#           python src/label_conversion/conversion_openlabel_to_kitti.py --root-dir /home/user/tum_traffic_intersection_dataset_split \
#                                                                        --out-dir /home/user/tum_traffic_intersection_dataset_split_kitti_format \
#                                                                        --format num


class OpenLABEL2KITTIConverter(object):
    """OpenLABEL to KITTI label format converter.
       This class serves as the converter to change the OpenLABEL format to KITTI format.
    """

    def __init__(self, splits, root_dir, out_dir, file_name_format):
        """
        Args:
            splits list[(str)]: Contains the different splits
            root_dir (str): Input folder path to OpenLABEL labels.
            out_dir (str): Output folder path to save data in KITTI format.
            file_name_format (str): Output file name of the converted file
        """

        self.splits = splits
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.file_name_format = file_name_format

        self.map_version_to_dir = {
            'training': 'train',
            'validation': 'val',
            'testing': 'test'
        }

        self.map_tum_traffic_to_kitti_sensors = {
            's110_lidar_ouster_north': 'velodyne_1',
            's110_lidar_ouster_south': 'velodyne_2',
            's110_camera_basler_south1_8mm': 'image_1',
            's110_camera_basler_south2_8mm': 'image_2'
        }

        self.train_set = []
        self.val_set = []
        self.test_set = []

        self.map_set_to_dir_idx = {
            'training': 0,
            'validation': 1,
            'testing': 2
        }

        self.imagesets = {
            'training': self.train_set,
            'validation': self.val_set,
            'testing': self.test_set
        }

        self.occlusion_map = {
            'NOT_OCCLUDED': 0,
            'PARTIALLY_OCCLUDED': 1,
            'MOSTLY_OCCLUDED': 2
        }

    def convert(self):

        print('Start converting ...')
        num_frames_training = -1
        for split in self.splits:
            for sensor_type in tqdm(
                    os.listdir(os.path.join(self.root_dir, self.map_version_to_dir[split], 'point_clouds'))):
                print(f'Converting split: {split}...')

                point_cloud_file_list = sorted(
                    glob(os.path.join(self.root_dir, self.map_version_to_dir[split], 'point_clouds', sensor_type, '*')))
                if split == "training":
                    num_frames_training = len(point_cloud_file_list)
                for file_idx, input_file_path_point_cloud in tqdm(enumerate(point_cloud_file_list)):
                    if split == "validation":
                        file_idx = file_idx + num_frames_training
                    out_file_name_no_ext = self.format_file_name(file_idx, input_file_path_point_cloud)
                    point_cloud_out_dir = f'{self.out_dir}/{self.map_version_to_dir[split]}/{self.map_tum_traffic_to_kitti_sensors[sensor_type]}'
                    os.makedirs(point_cloud_out_dir, exist_ok=True)
                    # self.save_point_cloud(input_file_path_point_cloud,
                    #                       os.path.join(point_cloud_out_dir, f'{out_file_name_no_ext}.bin'))

                    input_file_path_label = input_file_path_point_cloud.replace('point_clouds',
                                                                                'labels_point_clouds').replace('pcd',
                                                                                                               'json')
                    # projection matrix from s110_lidar_ouster_south to s110_camera_basler_south2_8mm
                    projection_matrix_velo_1_to_cam_2 = np.array(
                        [[1318.95273325, -859.15213894, -289.13390611, 11272.03223502],
                         [90.01799314, -2.9727517, -1445.63809767, 585.78988153],
                         [0.876766, 0.344395, -0.335669, -7.26891]])
                    # projection matrix from s110_lidar_ouster_south to s110_camera_basler_south1_8mm
                    projection_matrix_velo_1_to_cam_1 = np.array([
                        [7.04216073e02, -1.37317442e03, -4.32235765e02, -2.03369364e04],
                        [-9.28351327e01, -1.77543929e01, -1.45629177e03, 9.80290034e02],
                        [8.71736000e-01, -9.03453000e-02, -4.81574000e-01, -2.58546000e00],
                    ])
                    # projection matrix from s110_lidar_ouster_north to s110_camera_basler_south2_8mm
                    projection_matrix_velo_2_to_cam_2 = np.array(
                        [[1.31895273e+03, -8.59152139e+02, -2.89133906e+02, 1.12720322e+04],
                         [9.00179931e+01, -2.97275170e+00, -1.44563810e+03, 5.85789882e+02],
                         [8.76766000e-01, 3.44395000e-01, -3.35669000e-01, -7.26891000e+00]]
                    )
                    # projection matrix from s110_lidar_ouster_north to s110_camera_basler_south1_8mm
                    projection_matrix_velo_2_to_cam_1 = np.array(
                        [[2.90064402e+02, -1.52265886e+03, -4.17082935e+02, -3.98031250e+02],
                         [-8.89628396e+01, 6.86258619e+00, -1.45178013e+03, 4.54220718e+02],
                         [8.18463845e-01, -3.28184922e-01, -4.71605571e-01, -1.82577035e-01]]
                    )
                    # camera images are already rectified
                    r0_rect = np.eye(3)
                    tr_velo_1_to_cam_2 = np.array(
                        [[0.641509, -0.766975, 0.0146997, 1.99131],
                         [-0.258939, -0.234538, -0.936986, 1.21464],
                         [0.722092, 0.597278, -0.349058, -1.50021],
                         [0.0, 0.0, 0.0, 1.0]]
                    )
                    tr_velo_1_to_cam_1 = np.array(
                        [[-0.0931837, -0.995484, 0.018077, -13.8309],
                         [-0.481033, 0.029117, -0.876219, 1.96067],
                         [0.871736, -0.0903453, -0.481574, -2.58546],
                         [0.0, 0.0, 0.0, 1.0]
                         ]
                    )
                    tr_velo_2_to_cam_2 = np.array(
                        [[0.37383, -0.927155, 0.0251845, 14.2181],
                         [-0.302544, -0.147564, -0.941643, 3.50648],
                         [0.876766, 0.344395, -0.335669, -7.26891],
                         [0.0, 0.0, 0.0, 1.0]
                         ]
                    )
                    tr_velo_2_to_cam_1 = np.array(
                        [[-0.374855, -0.926815, 0.0222604, -0.284537],
                         [-0.465575, 0.167432, -0.869026, 0.683219],
                         [0.8017, -0.336123, -0.494264, -0.837352],
                         [0.0, 0.0, 0.0, 1.0]
                         ]
                    )

                    # write calibration data to file
                    calib_out_dir = f'{self.out_dir}/{self.map_version_to_dir[split]}/calib'
                    os.makedirs(calib_out_dir, exist_ok=True)
                    self.save_calibration(os.path.join(calib_out_dir, f'{out_file_name_no_ext}.txt'),
                                          projection_matrix_velo_1_to_cam_2, projection_matrix_velo_1_to_cam_1,
                                          projection_matrix_velo_2_to_cam_2, projection_matrix_velo_2_to_cam_1,
                                          r0_rect, tr_velo_1_to_cam_2, tr_velo_1_to_cam_1, tr_velo_2_to_cam_2,
                                          tr_velo_2_to_cam_1)

                    if sensor_type == 's110_lidar_ouster_north':
                        label_dir_name = 'label_1'
                        # North lidar labels are associated with south1 camera (image_1)
                        label_projection_matrix = projection_matrix_velo_2_to_cam_1
                    elif sensor_type == 's110_lidar_ouster_south':
                        label_dir_name = 'label_2'
                        # South lidar labels are associated with south2 camera (image_2)
                        label_projection_matrix = projection_matrix_velo_1_to_cam_2
                    else:
                        raise ValueError(
                            'Sensor type not supported. Please choose between "s110_lidar_ouster_north" or "s110_lidar_ouster_south"')

                    label_out_dir = f'{self.out_dir}/{self.map_version_to_dir[split]}/{label_dir_name}'
                    os.makedirs(label_out_dir, exist_ok=True)
                    self.save_label(input_file_path_label,
                                    os.path.join(label_out_dir, f'{out_file_name_no_ext}.txt'),
                                    projection_matrix=label_projection_matrix)
                    self.imagesets[split].append(out_file_name_no_ext + '\n')
                    file_idx += 1
        for split in self.splits:
            for sensor_type in tqdm(os.listdir(os.path.join(self.root_dir, self.map_version_to_dir[split], 'images'))):
                image_file_list = sorted(
                    glob(os.path.join(self.root_dir, self.map_version_to_dir[split], 'images', sensor_type, '*')))
                for file_idx, input_file_path_image in tqdm(enumerate(image_file_list)):
                    if split == "validation":
                        file_idx = file_idx + num_frames_training
                    out_file_name_no_ext = self.format_file_name(file_idx, input_file_path_image)
                    image_out_dir = f'{self.out_dir}/{self.map_version_to_dir[split]}/{self.map_tum_traffic_to_kitti_sensors[sensor_type]}'
                    os.makedirs(image_out_dir, exist_ok=True)
                    self.save_image(input_file_path_image, os.path.join(image_out_dir, out_file_name_no_ext + '.jpg'))

        print('Creating ImageSets...')
        self.create_imagesets()
        print('\nFinished ...')

    def format_file_name(self, file_idx, input_file_path):
        """
        Create the specified file name convention
        Args:
            file_idx: Index of the file in the given split
            input_file_path: Input file path

        Returns: Specified file name without extension

        """
        if self.file_name_format == 'name':
            return os.path.basename(input_file_path).split('.')[0]
        else:
            return f'{str(file_idx).zfill(6)}'

    # @staticmethod
    # def save_point_cloud(input_file_path_point_cloud, output_file_path_point_cloud):
    #     """
    #     Converts file from .pcd to .bin
    #     Args:
    #         input_file_path_point_cloud: Input file path to .pcd file
    #         output_file_path_point_cloud: Output filepath to .bin file
    #     """
    #     point_cloud = pypcd.PointCloud.from_path(input_file_path_point_cloud)
    #     np_x = np.array(point_cloud.pc_data['x'], dtype=np.float32)
    #     np_y = np.array(point_cloud.pc_data['y'], dtype=np.float32)
    #     np_z = np.array(point_cloud.pc_data['z'], dtype=np.float32)
    #     np_i = np.array(point_cloud.pc_data['intensity'], dtype=np.float32) / 256
    #     bin_format = np.column_stack((np_x, np_y, np_z, np_i))
    #     bin_format.tofile(os.path.join(output_file_path_point_cloud))

    @staticmethod
    def compute_2d_box_from_3d(cuboid_val, projection_matrix, img_width=1920, img_height=1200):
        """
        Projects a 3D cuboid to the image plane and returns the axis-aligned 2D bounding box.

        Args:
            cuboid_val: list [x, y, z, qx, qy, qz, qw, length, width, height] in LiDAR frame
            projection_matrix: 3×4 numpy array mapping LiDAR coords to image coords
            img_width: image width in pixels (default 1920)
            img_height: image height in pixels (default 1200)

        Returns:
            [xmin, ymin, xmax, ymax] clipped to image bounds, or [0, 0, 0, 0] if box is
            entirely outside the image or behind the camera.
        """
        x, y, z = cuboid_val[0], cuboid_val[1], cuboid_val[2]
        qx, qy, qz, qw = cuboid_val[3], cuboid_val[4], cuboid_val[5], cuboid_val[6]
        l, w, h = cuboid_val[7], cuboid_val[8], cuboid_val[9]

        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)],
        ])

        # 8 corners in the object frame
        corners_obj = np.array([
            [ l / 2,  w / 2,  h / 2],
            [ l / 2,  w / 2, -h / 2],
            [ l / 2, -w / 2,  h / 2],
            [ l / 2, -w / 2, -h / 2],
            [-l / 2,  w / 2,  h / 2],
            [-l / 2,  w / 2, -h / 2],
            [-l / 2, -w / 2,  h / 2],
            [-l / 2, -w / 2, -h / 2],
        ])

        # Transform corners to LiDAR frame
        corners_lidar = (R @ corners_obj.T).T + np.array([x, y, z])

        # Project to image using 3×4 projection matrix
        ones = np.ones((8, 1))
        corners_hom = np.hstack([corners_lidar, ones])
        proj = (projection_matrix @ corners_hom.T).T

        # Discard points behind the camera (depth <= 0)
        valid = proj[:, 2] > 0
        if not np.any(valid):
            return [0, 0, 0, 0]
        proj = proj[valid]
        px = proj[:, 0] / proj[:, 2]
        py = proj[:, 1] / proj[:, 2]

        xmin = float(np.clip(np.min(px), 0, img_width - 1))
        ymin = float(np.clip(np.min(py), 0, img_height - 1))
        xmax = float(np.clip(np.max(px), 0, img_width - 1))
        ymax = float(np.clip(np.max(py), 0, img_height - 1))

        if xmin >= xmax or ymin >= ymax:
            return [0, 0, 0, 0]
        return [xmin, ymin, xmax, ymax]

    def save_label(self, input_file_path_label, output_file_path_label,
                   projection_matrix=None, img_width=1920, img_height=1200):
        """
        Converts OpenLABEL format to KITTI label format
        Args:
            input_file_path_label: Input file path to .json label file
            output_file_path_label: Output file path to .txt label file
            projection_matrix: 3×4 numpy array for projecting 3D LiDAR points to the
                               corresponding camera image plane. When provided, real 2D
                               bounding boxes are computed by projecting the 8 corners of
                               each 3D cuboid; otherwise all 2D boxes are written as zeros.
            img_width: width of the camera image in pixels (default 1920)
            img_height: height of the camera image in pixels (default 1200)
        """
        # read json file
        lines = []
        json_file = open(input_file_path_label)
        json_data = json.load(json_file)
        for frame_id, label_json in json_data['openlabel']['frames'].items():
            for track_uuid, label_object in label_json['objects'].items():
                category = label_object['object_data']['type']
                # Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries
                # NOTE: truncation set to 0.0 as we do not have any information about it
                truncated = 0.00
                # 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
                # NOTE: occlusion set to 3 as we do not have any information about it
                occluded = 3
                for item in label_object['object_data']['cuboid']['attributes']['text']:
                    if item['name'] == 'occlusion_level':
                        occluded = self.occlusion_map[item['val']]
                # Observation angle of object, ranging [-pi..pi]
                # NOTE: observation angle (alpha) set to 0.0 as we do not have any information about it
                alpha = 0.00
                cuboid = label_object['object_data']['cuboid']['val']
                x_center = cuboid[0]
                y_center = cuboid[1]
                z_center = cuboid[2]
                length = cuboid[7]
                width = cuboid[8]
                height = cuboid[9]
                _, _, yaw = self.quaternion_to_euler(cuboid[3], cuboid[4], cuboid[5], cuboid[6])
                if projection_matrix is not None:
                    bounding_box = self.compute_2d_box_from_3d(
                        cuboid, projection_matrix, img_width, img_height)
                else:
                    bounding_box = [0, 0, 0, 0]
                line = f"{category} {round(truncated, 2)} {occluded} {round(alpha, 2)} " + \
                       f"{round(bounding_box[0], 2)} {round(bounding_box[1], 2)} {round(bounding_box[2], 2)} " + \
                       f"{round(bounding_box[3], 2)} {round(height, 2)} {round(width, 2)} {round(length, 2)} " + \
                       f"{round(x_center, 2)} {round(y_center, 2)} {round(z_center, 2)} {round(yaw, 2)}\n"
                lines.append(line)
        fp_label = open(output_file_path_label, 'a')
        fp_label.writelines(lines)
        fp_label.close()

    @staticmethod
    def quaternion_to_euler(q0, q1, q2, q3):
        """
        Converts quaternions to euler angles using unique transformation via atan2

        Returns: roll, pitch and yaw

        """
        roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
        pitch = np.arcsin(2 * (q0 * q2 - q3 * q1))
        yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
        return roll, pitch, yaw

    def create_imagesets(self):
        """
        Creates the ImageSets train.txt, val.txt, trainval.txt and test.txt each containing corresponding files
        """
        os.makedirs(os.path.join(self.out_dir, 'ImageSets'))

        with open(os.path.join(self.out_dir, 'ImageSets', 'train.txt'), 'w') as file:
            file.writelines(self.train_set)

        with open(os.path.join(self.out_dir, 'ImageSets', 'val.txt'), 'w') as file:
            file.writelines(self.val_set)

        with open(os.path.join(self.out_dir, 'ImageSets', 'trainval.txt'), 'w') as file:
            file.writelines(self.train_set)
            file.writelines(self.val_set)

        with open(os.path.join(self.out_dir, 'ImageSets', 'test.txt'), 'w') as file:
            file.writelines(self.test_set)

    def save_image(self, input_file_path_image, output_file_path_image):
        """
        Saves image to new location as .jpg
        Args:
            input_file_path_image: Input file path to image
            output_file_path_image: Output file path to image
        """
        img = cv2.imread(input_file_path_image)
        cv2.imwrite(output_file_path_image, img)

    def save_calibration(self, output_file_path, projection_matrix_velo_1_to_cam_2, projection_matrix_velo_1_to_cam_1,
                         projection_matrix_velo_2_to_cam_2, projection_matrix_velo_2_to_cam_1, r0_rect,
                         tr_velo_1_to_cam_2, tr_velo_1_to_cam_1, tr_velo_2_to_cam_2, tr_velo_2_to_cam_1):
        """
        Saves calibration to new location as .txt
        :param output_file_path: Output file path to calibration
        :param projection_matrix_velo_1_to_cam_2: Projection matrix from s110_lidar_ouster_south LiDAR to s110_camera_basler_south2_8mm camera
        :param projection_matrix_velo_1_to_cam_1: Projection matrix from s110_lidar_ouster_south LiDAR to s110_camera_basler_south1_8mm camera
        :param projection_matrix_velo_2_to_cam_2: Projection matrix from s110_lidar_ouster_north LiDAR to s110_camera_basler_south2_8mm camera
        :param projection_matrix_velo_2_to_cam_1: Projection matrix from s110_lidar_ouster_north LiDAR to s110_camera_basler_south1_8mm camera
        :param r0_rect: Rectification matrix
        :param tr_velo_1_to_cam_2: Transformation matrix from s110_lidar_ouster_south LiDAR to s110_camera_basler_south2_8mm camera
        :param tr_velo_1_to_cam_1: Transformation matrix from s110_lidar_ouster_south LiDAR to s110_camera_basler_south1_8mm camera
        :param tr_velo_2_to_cam_2: Transformation matrix from s110_lidar_ouster_north LiDAR to s110_camera_basler_south2_8mm camera
        :param tr_velo_2_to_cam_1: Transformation matrix from s110_lidar_ouster_north LiDAR to s110_camera_basler_south1_8mm camera
        """
        lines = []
        lines.append('P0: ' + ' '.join(map(str, projection_matrix_velo_1_to_cam_2.reshape(12).tolist())) + '\n')
        lines.append('P1: ' + ' '.join(map(str, projection_matrix_velo_1_to_cam_1.reshape(12).tolist())) + '\n')
        lines.append('P2: ' + ' '.join(map(str, projection_matrix_velo_2_to_cam_2.reshape(12).tolist())) + '\n')
        lines.append('P3: ' + ' '.join(map(str, projection_matrix_velo_2_to_cam_1.reshape(12).tolist())) + '\n')
        lines.append('R0_rect: ' + ' '.join(map(str, r0_rect.reshape(9).tolist())) + '\n')
        lines.append('Tr_velo_to_cam_0: ' + ' '.join(map(str, tr_velo_1_to_cam_2.reshape(4, 4)[:3, :].reshape(12).tolist())) + '\n')
        lines.append('Tr_velo_to_cam_1: ' + ' '.join(map(str, tr_velo_1_to_cam_1.reshape(4, 4)[:3, :].reshape(12).tolist())) + '\n')
        lines.append('Tr_velo_to_cam_2: ' + ' '.join(map(str, tr_velo_2_to_cam_2.reshape(4, 4)[:3, :].reshape(12).tolist())) + '\n')
        lines.append('Tr_velo_to_cam_3: ' + ' '.join(map(str, tr_velo_2_to_cam_1.reshape(4, 4)[:3, :].reshape(12).tolist())) + '\n')
        fp_calib = open(output_file_path, 'w')
        fp_calib.writelines(lines)
        fp_calib.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--root-dir',
        type=str,
        default='/home/user/tum_traffic_intersection_dataset_split',
        help='Specify the root folder path to the dataset containing the train and val folder.')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='/home/user/tum_traffic_intersection_dataset_split_kitti_format',
        required=False,
        help='Output directory for converted dataset')
    parser.add_argument(
        '--file-name-format',
        type=str,
        default='num',
        required=False,
        choices=['name', 'num'],
        help="Specify whether to keep original filenames or convert to numbering (e.g. 000000.txt) to be mmdetection3d/pcdet compatible"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    splits = ['training', 'validation']
    converter = OpenLABEL2KITTIConverter(
        splits=splits,
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        file_name_format=args.file_name_format
    )
    converter.convert()