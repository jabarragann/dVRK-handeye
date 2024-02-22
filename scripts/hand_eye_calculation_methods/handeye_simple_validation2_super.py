from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import pandas as pd
from dvrk_handeye.opencv_utils import draw_axis
from dvrk_handeye.DataLoading import load_images_data, load_joint_data_super
from dvrk_handeye.PSM_fk import compute_FK

# fmt: off
# # Camera intrinsics
# mtx =   [1767.7722 ,    0.     ,  529.11477,
#             0.     , 1774.33579,  510.58841,
#             0.     ,    0.     ,    1.       ]
# # Camera distortion
# dist = [-0.337317, 0.500592, 0.001082, 0.002775, 0.000000]

# # Output of the hand-eye calibration
# cam_T_base = [[-0.722624032680848,    0.5730602138325281, -0.3865443036888354,  -0.06336454384831497], 
#               [ 0.6870235345618597,   0.533741788848803,  -0.49307034568569336, -0.15304205999332426], 
#               [-0.07624414961292797, -0.621869515379792,  -0.7794004974922103,   0.0664797333995826], 
#               [0.0, 0.0, 0.0, 1.0]]

## SUPER FRAMEWORK
mtx =  [ 1.6435516401714499e+03, 0.,                     8.3384217485027705e+02, 
         0.,                     1.6328119551437901e+03, 6.5798955473076103e+02, 
         0.,                     0.,                     1. ]
dist = [ -4.0444238705587998e-01, 5.8161897902897197e-01,
       -4.9797819387316098e-03, 2.3217574337593299e-03,
       -2.1547479006608700e-01 ]
cam_T_base =  [[-7.20518834e-01, -6.93364279e-01, 9.92907158e-03,  9.38059998e+01],
                [-4.53949588e-01,  4.60807033e-01, -7.62618286e-01, -5.64430466e+01],
                [ 5.24196892e-01, -5.53988136e-01, -6.46772575e-01,  7.62134552e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],]

mtx = np.array(mtx).reshape(3, 3)
dist = np.array(dist)
cam_T_base = np.array(cam_T_base)
cam_T_base[:3,3] = cam_T_base[:3,3] / 1000 # convert to meters
base_T_cam = np.linalg.inv(cam_T_base)

# fmt: on


def draw_pose_on_img(
    img: np.ndarray, robot_pose: np.ndarray, size: float = 0.001
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
    color = (0, 255, 255)
    img = cv2.line(img, point1, point2, color, thickness)
    return img


def create_trans_matrix(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def draw_instrument_skeleton(
    img: np.ndarray,
    measured_jp: np.ndarray,
    measured_cp: np.ndarray,
    img_idx: int = -1,
    debug: bool = False,
) -> np.ndarray:

    base_T_gripper = measured_cp
    base_T_gripper_from_fk = compute_FK(measured_jp, 7)
    base_T_wrist_pitch = compute_FK(measured_jp, 5)
    # create pose along shaft
    offset_T = create_trans_matrix(-0.05, 0, 0)
    base_T_shaft = base_T_wrist_pitch @ offset_T

    # img, gripper_in_img = draw_pose_on_img(img, base_T_gripper)
    img, gripper_from_fk_in_img = draw_pose_on_img(img, base_T_gripper_from_fk)
    img, pitch_axis_in_img = draw_pose_on_img(img, base_T_wrist_pitch)
    # img, shaft_in_img = draw_pose_on_img(img, base_T_shaft)

    # img = draw_line_on_img(gripper_from_fk_in_img, pitch_axis_in_img, img)
    # img = draw_line_on_img(pitch_axis_in_img, shaft_in_img, img)

    if debug:
        print(f"{base_T_gripper}")
        print(f"image resolution {img.shape}")
        # print(f"location of gripper in img {gripper_in_img}")
        print(f"location of gripper from fk in img {gripper_from_fk_in_img}")
        print(f"location of pitch axis in img {pitch_axis_in_img}")
        # print(f"location of shaft in img {shaft_in_img}")

    # try:
    #     assert gripper_in_img == gripper_from_fk_in_img
    # except AssertionError:
    #     print(
    #         f"Discrepancies between fk model ({gripper_in_img}) and measured_cp ({gripper_from_fk_in_img}) in img {img_idx}"
    #     )

    return img


def single_img_validation():
    # raw images
    root_path = Path("datasets/20240213_212626_raw_dataset_handeye_raw_img_local")

    # rectified images
    # root_path = Path("datasets/20240213_212744_raw_dataset_handeye_rect_img_local")

    root_path = Path("datasets/20240220_220519_super_graps1")

    idx = 31
    img = load_images_data(root_path, idx)

    # measured_cp = load_poses_data(root_path)
    measured_jp = load_joint_data_super(root_path)

    img = draw_instrument_skeleton(img, measured_jp[idx], None, idx, debug=True)

    window_name = "Resized_Window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 360)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def multi_img_validation():
    from dvrk_handeye.opencv_utils import VideoWriter

    # raw images
    root_path = Path("datasets/20240213_212626_raw_dataset_handeye_raw_img_local")

    # rectified images
    # root_path = Path("datasets/20240213_212744_raw_dataset_handeye_rect_img_local")

    root_path = Path("datasets/20240220_220519_super_graps1")

    file_name = root_path.name + ".mp4"
    video_writer = VideoWriter(
        output_dir="temp",
        width=1920,
        height=1080,
        fps=3,
        file_name=file_name,
    )

    idx = 31
    img = load_images_data(root_path, idx)
    # measured_cp = load_poses_data(root_path)
    measured_jp = load_joint_data_super(root_path)

    with video_writer as writer:
        for i in range(0, measured_jp.shape[0] - 20):
            img = load_images_data(root_path, i)
            img = draw_instrument_skeleton(img, measured_jp[i], None, i, debug=False)
            writer.write_frame(i, img)


if __name__ == "__main__":
    # single_img_validation()

    multi_img_validation()
