from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from dvrk_handeye.opencv_utils import draw_axis

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


def load_images_data(root_path: Path, idx: int):
    img_path = root_path / "imgs" / "left" / f"camera_l_{idx:05d}.jpeg"

    left_img = cv2.imread(str(img_path))
    return left_img


def load_poses_data(root_path: Path):
    file_path = root_path / "pose_data.csv"
    pose_data = pd.read_csv(file_path).values
    pose_data = pose_data[:, 1:]  # remove index column
    pose_data = pose_data.reshape(-1, 4, 4)

    return pose_data


def main():
    # raw images
    # root_path = Path("datasets/20240213_212626_raw_dataset_handeye_raw_img_local")

    # rectified images
    root_path = Path("datasets/20240213_212744_raw_dataset_handeye_rect_img_local")

    idx = 68
    img = load_images_data(root_path, idx)

    pose_data = load_poses_data(root_path)
    base_T_gripper = pose_data[idx]

    pose = cam_T_base @ base_T_gripper
    tvec = pose[:3, 3]
    rvec = cv2.Rodrigues(pose[:3, :3])[0]

    points_3d = np.array([[[0, 0, 0]]], np.float32)
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, mtx, dist)

    # Anotate image with gripper pose
    img = draw_axis(img, mtx, dist, pose, size=0.01)
    points_2d = tuple(points_2d.astype(np.int32)[0, 0])
    img = cv2.circle(img, points_2d, 10, (255, 255, 255), -1)

    print(f"image resolution {img.shape}")
    print(f"center of axis {points_2d}")

    window_name = "Resized_Window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
