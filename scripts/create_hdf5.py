import numpy as np
import pickle
import cv2
import h5py
import numpy as np
from pathlib import Path


def process_marker_pose(tvec: np.ndarray, rvec: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = tvec[:, 0]
    rot_matrix = cv2.Rodrigues(rvec)[0]
    pose[:3, :3] = rot_matrix

    return pose


hdf_path = Path("data/test.hdf5")
window_name = "marker_detected"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)

N = 32
with h5py.File(hdf_path, "w") as hdf_file:
    robot_poses = hdf_file.create_dataset("robot_poses", (4, 4, N), dtype="f")
    marker_poses = hdf_file.create_dataset("marker_poses", (4, 4, N), dtype="f")

    images_group = hdf_file.create_group("images")
    robot_poses.attrs["Shape"] = "4x4xN where N is the number of data points"
    marker_poses.attrs["Shape"] = "4x4xN where N is the number of data points"

    for i in range(N):
        file_p = Path(f"data/handeye_data_v1/data_{i:02d}.pkl")
        with open(file_p, "rb") as file:
            data_dict = pickle.load(file)

        robot_poses[:, :, i] = data_dict["robot_cp"]
        marker_poses[:,:,i] = process_marker_pose(data_dict["marker_tvec"], data_dict["marker_rvec"])

        dataset_name = f"image_{i}"
        images_group.create_dataset(dataset_name, data=data_dict["img"])

        cv2.imshow(window_name, data_dict["img"])
        cv2.waitKey(0)

    cv2.destroyAllWindows()
