from pathlib import Path
from dvrk_handeye.DataLoading import (
    load_poses_data,
    load_joint_data,
)
from dvrk_handeye.PSM_fk import compute_FK, round_mat
import pandas as pd
import numpy as np

if __name__ == "__main__":

    root_path = Path("./datasets/20240213_212626_raw_dataset_handeye_raw_img_local")
    # root_path = Path("./datasets/20240213_212744_raw_dataset_handeye_rect_img_local/")

    measured_jp = load_joint_data(root_path)
    measured_cp = load_poses_data(root_path)

    errors = 0
    rot_tolerance = 1e-2
    trans_tolerance = 1e-4

    print("compare FK against robot's measured_cp...")
    for idx in range(measured_jp.shape[0]):
        p1 = compute_FK(measured_jp[idx], 7)

        try:
            assert np.allclose(measured_cp[idx, :3, :3], p1[:3, :3], atol=rot_tolerance)
            assert np.allclose(
                measured_cp[idx, :3, 3].squeeze(),
                p1[:3, 3].squeeze(),
                atol=trans_tolerance,
            )
        except AssertionError:
            print(f"\nPose {idx} has errors bigger than tolerances")

            print("rotational error")
            print(measured_cp[idx, :3, :3] - p1[:3, :3], "\n")
            print(abs(measured_cp[idx, :3, :3] - p1[:3, :3]) < rot_tolerance, "\n")

            print("translational error")
            print(measured_cp[idx, :3, 3].squeeze() - p1[:3, 3].squeeze(), "\n")
            print(
                abs(measured_cp[idx, :3, 3].squeeze() - p1[:3, 3].squeeze())
                < trans_tolerance,
                "\n",
            )

            errors += 1

    print(f"There are {errors} measurements with high errors")
