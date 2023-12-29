import h5py
from dvrk_handeye.PoseEstimation import Batch_Processing


def main():
    with h5py.File("data/collection1.hdf5", "r") as hdf_file:
        robot_poses = hdf_file["robot_poses"][()]
        marker_poses = hdf_file["marker_poses"][()]

    X, Y, Y_est_check, error_stats = Batch_Processing.pose_estimation(
        A=robot_poses, B=marker_poses
    )

    print("X:\n", X)
    print("Y:\n", Y)
    print("Y_est_check:\n", Y_est_check)
    print("error_stats:\n", error_stats)

    with h5py.File("data/collection1.hdf5", "a") as hdf_file:
        handeye_group = hdf_file.create_group("hand_eye_calibration")
        handeye_group.create_dataset("X", data=X)
        handeye_group.create_dataset("Y", data=Y)

        handeye_group.create_dataset("Y_est_check", data=Y_est_check)
        handeye_group.attrs["mean_error"] = error_stats[0, 0]
        handeye_group.attrs["std_error"] = error_stats[1, 0]


if __name__ == "__main__":
    main()
