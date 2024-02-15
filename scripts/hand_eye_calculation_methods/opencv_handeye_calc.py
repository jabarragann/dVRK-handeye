import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from dvrk_handeye.VisionTracker import VisionTracker


# fmt: off
mtx = np.array( [ [1747.03274843401, 0.0, 521.0492603841079], [0.0, 1747.8694061586648, 494.32395180615987], [0.0, 0.0, 1.0], ])
dist = np.array( [ -0.33847195608374453, 0.16968704500434714, 0.0007293228134352138, 0.005422675750927001, 0.9537762252401928, ])
marker_size = 0.01
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

    vision_tracker = VisionTracker(mtx, dist, marker_size)
    root_path = Path("datasets/20240213_212744_raw_dataset_handeye_rect_img_local")

    img_path = root_path / "imgs" / "left"
    total_images = len(list(img_path.glob("*.jpeg")))
    print(f"Total images: {total_images}")

    last_img_to_use = 102
    for i in range(last_img_to_use):
        img = load_images_data(root_path, i)

        corners, ids, rejectedImgPoints = vision_tracker.detect_markers(img)

        if len(corners) > 0:
            print(f"Image {i}: Found markers")
        else:
            print(f"Image {i}: No markers found")

    # idx = 30
    # img = load_images_data(root_path, idx)

    # pose_data = load_poses_data(root_path)

    # pose = Y_inv @ pose_data[idx]
    # tvec = pose[:3, 3]
    # rvec = cv2.Rodrigues(pose[:3, :3])[0]

    # points_3d = np.array([[[0, 0, 0]]], np.float32)
    # points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, mtx, dist)

    # print(points_2d)
    # print(pose_data[idx])

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
