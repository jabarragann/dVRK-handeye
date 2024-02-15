import cv2
from pathlib import Path
from cv2 import aruco
from dataclasses import dataclass
import numpy as np
from dvrk_handeye.VisionTracker import VisionTracker

# fmt: off
mtx = np.array( [ [1747.03274843401, 0.0, 521.0492603841079], [0.0, 1747.8694061586648, 494.32395180615987], [0.0, 0.0, 1.0], ])
dist = np.array( [ -0.33847195608374453, 0.16968704500434714, 0.0007293228134352138, 0.005422675750927001, 0.9537762252401928, ])
marker_size = 0.01
# fmt: on


def detect_aruco_markers():
    vision_tracker = VisionTracker(mtx, dist, marker_size)
    path = Path(
        "./datasets/20240213_212626_raw_dataset_handeye_raw_img_local/imgs/left/camera_l_00005.jpeg"
    )
    # path = Path(
    #     "./datasets/20240213_212744_raw_dataset_handeye_rect_img_local/imgs/left/camera_l_00005.jpeg"
    # )
    path = Path("./data/other/puppy.jpg")

    img = cv2.imread(str(path))
    corners, ids, rejectedImgPoints = vision_tracker.detect_markers(img)

    if len(corners) > 0:
        rvec, tvec, marker_points = vision_tracker.estimate_pose(corners[0])
        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
        frame_markers = cv2.drawFrameAxes(frame_markers, mtx, dist, rvec, tvec, 0.01)

        print(rvec)
        print(tvec)
        print(corners)
        # print(marker_points)

        window_name = "Resized_Window"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        cv2.imshow(window_name, frame_markers)
        # cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No markers found")


if __name__ == "__main__":
    detect_aruco_markers()
