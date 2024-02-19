from dataclasses import dataclass
from pathlib import Path
from typing import Union
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


@dataclass
class VideoWriter:
    output_dir: Union[Path, str]
    width: int
    height: int
    fps: int
    file_name: str = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.file_name is None:
            self.full_path = self.output_dir / f"output.mp4"
        else:
            self.full_path = self.output_dir / self.file_name

        self.full_path = str(self.full_path)
        self.fourcc = cv2.VideoWriter_fourcc(*"MPEG")

        self.out = cv2.VideoWriter(
            self.full_path, self.fourcc, self.fps, (self.width, self.height)
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.out.release()

    def write_frame(self, idx: int, frame: np.ndarray):
        self.out.write(frame)
