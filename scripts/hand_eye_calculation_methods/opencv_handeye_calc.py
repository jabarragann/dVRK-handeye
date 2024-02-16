import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from dvrk_handeye.VisionTracker import VisionTracker
from cv2 import aruco


# fmt: off
mtx = np.array( [ [1747.03274843401, 0.0, 521.0492603841079], [0.0, 1747.8694061586648, 494.32395180615987], [0.0, 0.0, 1.0], ])
dist = np.array( [ -0.33847195608374453, 0.16968704500434714, 0.0007293228134352138, 0.005422675750927001, 0.9537762252401928, ])
marker_size = 0.01
# fmt: on


def load_images_data(root_path: Path, idx: int):
    img_path = root_path / "imgs" / "left" / f"camera_l_{idx:05d}.jpeg"

    left_img = cv2.imread(str(img_path))
    return left_img


def load_poses_data(root_path: Path) -> np.ndarray:
    file_path = root_path / "pose_data.csv"
    pose_data = pd.read_csv(file_path).values
    pose_data = pose_data[:, 1:]  # remove index column
    pose_data = pose_data.reshape(-1, 4, 4)

    return pose_data


def to_homogeneous(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.concatenate([rot, t], axis=1)
    T = np.vstack([T, [0, 0, 0, 1]])
    return T


def main():

    vision_tracker = VisionTracker(mtx, dist, marker_size)
    root_path = Path("datasets/20240213_212744_raw_dataset_handeye_rect_img_local")

    img_path = root_path / "imgs" / "left"
    total_images = len(list(img_path.glob("*.jpeg")))

    robot_poses = load_poses_data(root_path)

    print(f"Total images: {total_images}")
    print(f"Total poses: {robot_poses.shape[0]}")

    last_img_to_use = 102

    cam_T_marker_rot = []
    cam_T_marker_t = []
    marker_T_cam_rot = []
    marker_T_cam_t = []
    gripper_T_base_rot = []
    gripper_T_base_t = []
    base_T_gripper_rot = []
    base_T_gripper_t = []
    imgs = []
    for i in range(last_img_to_use):
        img = load_images_data(root_path, i)

        corners, ids, rejectedImgPoints = vision_tracker.detect_markers(img)

        if len(corners) > 0:
            print(f"Image {i}: Found markers")
            rvec, tvec, marker_points = vision_tracker.estimate_pose(corners[0])
            marker_R = cv2.Rodrigues(rvec)[0]
            
            imgs.append(img)

            cam_T_marker = to_homogeneous(marker_R, tvec[0].T)
            cam_T_marker_rot.append(marker_R)
            cam_T_marker_t.append(tvec)

            marker_T_cam = np.linalg.inv(cam_T_marker)
            marker_T_cam_rot.append(marker_T_cam[:3, :3])
            marker_T_cam_t.append(marker_T_cam[:3, 3])

            gripper_T_base = np.linalg.inv(robot_poses[i])
            gripper_T_base_rot.append(gripper_T_base[:3, :3])
            gripper_T_base_t.append(gripper_T_base[:3, 3])

            base_T_gripper_rot.append(robot_poses[i][:3, :3])
            base_T_gripper_t.append(robot_poses[i][:3, 3])

        else:
            print(f"Image {i}: No markers found")

    base_T_cam_rot, base_T_cam_t = cv2.calibrateHandEye(
        gripper_T_base_rot,
        gripper_T_base_t,
        cam_T_marker_rot,
        cam_T_marker_t,
        method=cv2.CALIB_HAND_EYE_HORAUD,
    )

    gripper_T_marker_rot, gripper_T_marker_t = cv2.calibrateHandEye(
        base_T_gripper_rot,
        base_T_gripper_t,
        marker_T_cam_rot,
        marker_T_cam_t,
        method=cv2.CALIB_HAND_EYE_HORAUD,
    )

    print(f"Base_T_Cam_Rot {base_T_cam_rot}")
    print(f"Base_T_Cam_t {base_T_cam_t}")
    print(f"dist between base and cam {np.linalg.norm(base_T_cam_t)}")

    print("single image validation")
    base_T_cam = to_homogeneous(base_T_cam_rot, base_T_cam_t)
    gripper_T_marker = to_homogeneous(
        gripper_T_marker_rot, gripper_T_marker_t
    )

    print(f"base_T_cam \n{base_T_cam}")
    print(f"gripper_T_marker \n{gripper_T_marker}")

    cam_T_base = np.linalg.inv(base_T_cam)

    idx = 30
    test_img = load_images_data(root_path, idx)
    base_T_gripper = robot_poses[idx]

    # marker
    corners, ids, rejectedImgPoints = vision_tracker.detect_markers(img)
    rvec, tvec, marker_points = vision_tracker.estimate_pose(corners[0])
    rvec = cv2.Rodrigues(rvec)[0]
    cam_T_marker = to_homogeneous(rvec, tvec[0].T)

    # calc gripper_T_marker
    gripper_T_marker_2 = gripper_T_base @ base_T_cam @ cam_T_marker
    print(f"gripper_T_marker_2 \n{gripper_T_marker_2}")

    # The nonsense section
    pose = cam_T_base @ base_T_gripper
    tvec = pose[:3, 3]
    rvec = cv2.Rodrigues(pose[:3, :3])[0]
    points_3d = np.array([[[0, 0, 0]]], np.float32)
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, mtx, dist)

    print(points_2d)

    cv2.imshow("img", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
