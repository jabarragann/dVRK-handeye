import cv2
from pathlib import Path
from cv2 import aruco
from dataclasses import dataclass
import numpy as np

# fmt: off
mtx = np.array( [ [1747.03274843401, 0.0, 521.0492603841079], [0.0, 1747.8694061586648, 494.32395180615987], [0.0, 0.0, 1.0], ])
dist = np.array( [ -0.33847195608374453, 0.16968704500434714, 0.0007293228134352138, 0.005422675750927001, 0.9537762252401928, ])
marker_size = 0.01
# fmt: on


@dataclass
class VisionTracker:
    mtx: np.ndarray
    dist: np.ndarray
    marker_size: float = 0.01

    def __post_init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def detect_markers(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)
        return corners, ids, rejectedImgPoints

    def estimate_pose(self, single_marker_corners):
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
            single_marker_corners, self.marker_size, self.mtx, self.dist
        )
        return rvec, tvec, markerPoints


def detect_aruco_markers():
    vision_tracker = VisionTracker(mtx, dist, marker_size)
    path = Path("./data/marker_pictures/sample1.jpg")
    # path = Path("./data/puppy.jpg")
    img = cv2.imread(str(path))
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    corners, ids, rejectedImgPoints = vision_tracker.detect_markers(img)
    rvec, tvec, marker_points = vision_tracker.estimate_pose(corners[0])
    frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
    frame_markers = cv2.drawFrameAxes(frame_markers, mtx, dist, rvec, tvec, 0.01)

    print(rvec)
    print(tvec)
    print(corners)
    # print(marker_points)

    cv2.imshow("img", frame_markers)
    # cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_aruco_markers()
