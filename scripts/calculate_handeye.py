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


if __name__ == "__main__":
    main()
