from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import pandas as pd
from dvrk_handeye.opencv_utils import draw_axis
from dvrk_handeye.DataLoading import load_images_data, load_poses_data, load_joint_data
from dvrk_handeye.PSM_fk import compute_FK

# fmt: off
# Camera intrinsics
mtx =   [1767.7722 ,    0.     ,  529.11477,
            0.     , 1774.33579,  510.58841,
            0.     ,    0.     ,    1.       ]
# Camera distortion
dist = [-0.337317, 0.500592, 0.001082, 0.002775, 0.000000]

# Output of the hand-eye calibration
cam_T_base = [[-0.722624032680848,    0.5730602138325281, -0.3865443036888354,  -0.06336454384831497], 
              [ 0.6870235345618597,   0.533741788848803,  -0.49307034568569336, -0.15304205999332426], 
              [-0.07624414961292797, -0.621869515379792,  -0.7794004974922103,   0.0664797333995826], 
              [0.0, 0.0, 0.0, 1.0]]

mtx = np.array(mtx).reshape(3, 3)
dist = np.array(dist)
cam_T_base = np.array(cam_T_base)
base_T_cam = np.linalg.inv(cam_T_base)

# fmt: on


def draw_pose_on_img(
    img: np.ndarray, robot_pose: np.ndarray, size: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    draw pose on image

    Parameters
    ----------
    img : np.ndarray
    robot_pose : np.ndarray
        pose with respect to the robot base frame
    size : float, optional
        size of axis in img, by default 0.01

    Returns
    -------
    img: np.ndarray
        image with axis drawn
    points_2d: np.ndarray
        center of pose in pixel coordinates
    """

    pose = cam_T_base @ robot_pose
    tvec = pose[:3, 3]
    rvec = cv2.Rodrigues(pose[:3, :3])[0]

    points_3d = np.array([[[0, 0, 0]]], np.float32)
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, mtx, dist)

    # Anotate image with gripper pose
    img = draw_axis(img, mtx, dist, pose, size=0.01)
    points_2d = tuple(points_2d.astype(np.int32)[0, 0])
    img = cv2.circle(img, points_2d, 10, (255, 255, 255), -1)

    return img, points_2d


def draw_line_on_img(
    point1: Tuple[int], point2: Tuple[int], img: np.ndarray
) -> np.ndarray:
    thickness = 4
    color = (255, 255, 255)
    img = cv2.line(img, point1, point2, color, thickness)
    return img


def create_trans_matrix(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def main():
    # raw images
    root_path = Path("datasets/20240213_212626_raw_dataset_handeye_raw_img_local")

    # rectified images
    # root_path = Path("datasets/20240213_212744_raw_dataset_handeye_rect_img_local")

    idx = 30
    img = load_images_data(root_path, idx)

    measured_cp = load_poses_data(root_path)
    measured_jp = load_joint_data(root_path)

    base_T_gripper = measured_cp[idx]
    base_T_gripper_from_fk = compute_FK(measured_jp[idx], 7)
    base_T_wrist_pitch = compute_FK(measured_jp[idx], 5)

    offset_T = create_trans_matrix(-0.05, 0, 0)
    base_T_shaft = base_T_wrist_pitch @ offset_T

    img, gripper_in_img = draw_pose_on_img(img, base_T_gripper)
    img, gripper_from_fk_in_img = draw_pose_on_img(img, base_T_gripper_from_fk)
    img, pitch_axis_in_img = draw_pose_on_img(img, base_T_wrist_pitch)
    img, shaft_in_img = draw_pose_on_img(img, base_T_shaft)

    img = draw_line_on_img(gripper_in_img, pitch_axis_in_img, img)
    img = draw_line_on_img(pitch_axis_in_img, shaft_in_img, img)

    print(f"{base_T_gripper}")
    print(f"image resolution {img.shape}")
    print(f"location of gripper in img {gripper_in_img}")
    print(f"location of gripper from fk in img {gripper_from_fk_in_img}")
    print(f"location of pitch axis in img {pitch_axis_in_img}")
    print(f"location of shaft in img {shaft_in_img}")

    window_name = "Resized_Window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
