import numpy as np
import pickle
import cv2
import h5py
import numpy as np
from pathlib import Path

hdf_path = Path("data/test.hdf5")
window_name = "marker_detected"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)

with h5py.File(hdf_path, "w") as hdf_file:
    images_group = hdf_file.create_group("images")

    for i in range(2):
        file_p = Path(f"data/handeye_data_v1/data_{i:02d}.pkl")
        with open(file_p, "rb") as file:
            data_dict = pickle.load(file)

        dataset_name = f"image_{i}"
        images_group.create_dataset(dataset_name, data=data_dict["img"])

        cv2.imshow(window_name, data_dict["img"])
        cv2.waitKey(0)

    cv2.destroyAllWindows()
