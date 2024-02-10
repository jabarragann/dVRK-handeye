from enum import Enum
from dvrk_handeye.hdf5_saver.Hdf5Writer import Hdf5EntryConfig, Hdf5FullDatasetConfig

# fmt: off
_chunk = 100 
class HandEyeDVRKConfig(Enum):
    camera_l = ("camera_l", (_chunk, 480, 640, 3), (None, 480, 640, 3), "gzip")
    camera_r = ("camera_r", (_chunk, 480, 640, 3), (None, 480, 640, 3), "gzip")
    psm1_measured_cp = ("psm1_measured_cp", (_chunk, 7), (None, 7), "gzip")
    psm1_measured_jp = ("psm1_measured_jp", (_chunk, 6), (None, 6), "gzip")

# fmt: on


if __name__ == "__main__":

    selected_configs = [HandEyeDVRKConfig.camera_l, HandEyeDVRKConfig.camera_r]
    dataset_config = Hdf5FullDatasetConfig.create_from_enum_list(selected_configs)

    for config in dataset_config:
        print(config)
