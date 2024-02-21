from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import cv2
import numpy as np
from yaml import load, dump
from yaml import Loader, Dumper
import yaml


def reshape_list_to_numpy_array(data: list, rows: int, cols: int) -> np.ndarray:
    return np.array(data).reshape((rows, cols))


@dataclass
class YamlConfigFile:
    file: Path

    def __post_init__(self):
        self.file = Path(self.file)
        if not self.file.exists():
            raise FileNotFoundError(f"File {self.file} does not exist")

        with self.file.open("r") as file_handle:
            self.yaml_file_handler = yaml.load(file_handle, Loader=Loader)


@dataclass
class CameraCalibrationConfig(YamlConfigFile):
    img_width: int = field(init=False)
    img_height: int = field(init=False)
    camera_name: str = field(init=False)
    camera_matrix: np.ndarray = field(init=False)
    distortion_coefficients: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        self.camera_name = self.yaml_file_handler["camera_name"]

        self.camera_matrix = reshape_list_to_numpy_array(
            self.yaml_file_handler["camera_matrix"]["data"],
            self.yaml_file_handler["camera_matrix"]["rows"],
            self.yaml_file_handler["camera_matrix"]["cols"],
        )
        self.distortion_coefficients = reshape_list_to_numpy_array(
            self.yaml_file_handler["distortion_coefficients"]["data"],
            self.yaml_file_handler["distortion_coefficients"]["rows"],
            self.yaml_file_handler["distortion_coefficients"]["cols"],
        )

        self.img_width = self.yaml_file_handler["image_width"]
        self.img_height = self.yaml_file_handler["image_height"]


@dataclass
class HandEyeCalibrationConfig(YamlConfigFile):
    cam_T_base: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        trans = reshape_list_to_numpy_array(
            self.yaml_file_handler["robot_2_camera"]["translation"]["data"],
            self.yaml_file_handler["robot_2_camera"]["translation"]["rows"],
            self.yaml_file_handler["robot_2_camera"]["translation"]["cols"],
        )
        rot = self.load_rotation_component()

        self.cam_T_base = np.eye(4)
        self.cam_T_base[:3, :3] = rot
        self.cam_T_base[:3, 3] = trans

    def load_rotation_component(self) -> np.ndarray:

        rows = self.yaml_file_handler["robot_2_camera"]["rotation"]["rows"]
        cols = self.yaml_file_handler["robot_2_camera"]["rotation"]["cols"]
        data = self.yaml_file_handler["robot_2_camera"]["rotation"]["data"]

        if rows == 1:  # Axis angle representation
            data = np.array(data).reshape((rows, cols))
            data = cv2.Rodrigues(data)[0]
        elif rows == 3:
            data = np.array(data).reshape((rows, cols))
        else:
            raise ValueError(
                "robot_2_camera entry in yaml file has invalid rotation component. Must be 1x3 or 3x3."
            )

        return data


def main():

    root_path = Path("./datasets/20240213_212626_raw_dataset_handeye_raw_img_local")
    camera_config = CameraCalibrationConfig(root_path / "camera_calibration.yaml")
    hand_eye_config = HandEyeCalibrationConfig(root_path / "hand_eye.yaml")

    print(repr(camera_config))
    print(repr(hand_eye_config))


if __name__ == "__main__":
    main()
