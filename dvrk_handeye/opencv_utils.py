import numpy as np
import cv2


def draw_axis(
    img: np.ndarray, mtx: np.ndarray, dist: np.ndarray, pose: np.ndarray, size: int = 10
):

    s = size
    thickness = 2
    R, t = pose[:3, :3], pose[:3, 3]
    K = mtx

    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[s, 0, 0], [0, s, 0], [0, 0, s], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, dist)
    axisPoints = axisPoints.astype(int)

    print(f"center of axis {axisPoints[-1]}")

    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()),
        tuple(axisPoints[0].ravel()),
        (255, 0, 0),
        thickness,
    )
    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()),
        tuple(axisPoints[1].ravel()),
        (0, 255, 0),
        thickness,
    )

    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()),
        tuple(axisPoints[2].ravel()),
        (0, 0, 255),
        thickness,
    )
    return img
