from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
from dvrk_handeye.hdf5_saver.SyncRosClient import (
    AbstractSimulationClient,
    DatasetSample,
)
import os
import h5py
from enum import Enum
from enum import Enum
from typing import Dict, List, Tuple


@dataclass
class Hdf5EntryConfig:
    dataset_name: str
    chunk_shape: Tuple[int]
    max_shape: Tuple[int]
    compression: str

    def __post_init__(self):
        self.chunk_size = self.chunk_shape[0]


@dataclass
class Hdf5FullDatasetConfig(list):
    data_config: List[Hdf5EntryConfig]

    def __post_init__(self):
        for config in self.data_config:
            self.append(config)

        self._idx = 0

    @classmethod
    def create_from_enum_list(cls, enum_list: List[Enum]):
        init_list = []
        for enum in enum_list:
            init_list.append(Hdf5EntryConfig(*enum.value))

        return cls(init_list)

    def __getitem__(self, idx) -> Hdf5EntryConfig:
        return self.data_config[idx]

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self) -> Hdf5EntryConfig:
        if self._idx < len(self.data_config):
            config = self.data_config[self._idx]
            self._idx += 1
            return config
        else:
            raise StopIteration


@dataclass
class DataContainer:
    """
    Container capable of storing fixed defined number of data points
    """

    dataset_config: Hdf5FullDatasetConfig

    def __post_init__(self):
        self._internal_data_dict: Dict[Hdf5EntryConfig, np.ndarray] = dict()
        self.internal_idx = 0
        self.max_idx = self.dataset_config[0].chunk_size

        for config in self.dataset_config:
            self._internal_data_dict[config.dataset_name] = np.zeros(config.chunk_shape)

    def add_data(self, data_dict: dict):
        """
        Data is meant to be feeded one element at a time.
        """

        if self.internal_idx >= self.max_idx:
            raise ValueError("Chunk is full")

        for key, value in data_dict.items():
            if key not in self._internal_data_dict:
                raise ValueError(f"Dataset {key} not found in the container config")
            self._internal_data_dict[key][self.internal_idx] = value

        self.internal_idx += 1


@dataclass
class HDF5Writer:
    output_dir: Path
    dataset_config: Hdf5FullDatasetConfig
    file_name: str = None

    def __post_init__(self):
        # Member attributes
        self.file_path = None
        self.current_dataset_size = None
        self._internal_idx = 0

    def _create_path(self) -> Path:
        if not os.path.exists(self.output_dir):
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.file_name is not None:
            file_path = self.output_dir / self.file_name
        else:
            file_path = self.output_dir / (time.strftime("%Y%m%d_%H%M%S") + ".hdf5")

        return file_path

    def _query_permission_to_overwrite(self, file_path) -> bool:
        if self.file_path.exists():
            ans = input(
                f"HDF5 file {file_path}.already exists. Do you want to overwrite it? (y/n) "
            )

            if ans != "y":
                raise ValueError(
                    f"File {file_path} already exists. Please choose another name."
                )

    def _init_hdf5_file(self):

        self.file_path = self._create_path()
        self._query_permission_to_overwrite(self.file_path)
        self.hdf5_file_handler = h5py.File(self.file_path, "w")

        metadata = self.hdf5_file_handler.create_group("metadata")
        metadata.create_dataset(
            "README",
            data="All position information is in meters unless specified otherwise. \n"
            "Quaternion is a list in the order of [qx, qy, qz, qw]. \n"
            "Poses are defined to be T_world_obj. \n"
            "Depth in CV convention (corrected by extrinsic, T_cv_ambf). \n",
        )

        self.data_group = self.hdf5_file_handler.create_group("data")
        self.create_datasets()

    def create_datasets(self):
        self.datasets_dict: Dict[str, h5py.Dataset] = dict()
        self.current_dataset_size = self.dataset_config[0].chunk_size

        for config in self.dataset_config:
            self.datasets_dict[config.dataset_name] = self.data_group.create_dataset(
                config.dataset_name,
                shape=config.chunk_shape,
                maxshape=config.max_shape,
                compression=config.compression,
            )

    def write_chunk(self, data_container: DataContainer):
        idx = self._internal_idx
        s = self.current_dataset_size

        config: Hdf5EntryConfig
        for config in self.dataset_config:
            self.datasets_dict[config.dataset_name][idx * s : (idx + 1) * s, :] = (
                data_container._internal_data_dict[config.dataset_name]
            )

            new_size = list(config.chunk_shape)
            new_size[0] = s * (idx + 1)
            self.datasets_dict[config.dataset_name].resize(new_size)

        self._internal_idx += 1

    def __enter__(self):
        self._init_hdf5_file()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hdf5_file_handler.close()


def test1():
    from dvrk_handeye.hdf5_saver.custom_configs.hand_eye_dvrk_config import (
        HandEyeDVRKConfig,
    )

    print("test1")
    dataset_config = Hdf5FullDatasetConfig.create_from_enum_list(
        [HandEyeDVRKConfig.camera_l, HandEyeDVRKConfig.camera_r]
    )

    h5_writer = HDF5Writer(Path("temp"), dataset_config)

    data_container = DataContainer(dataset_config)

    for i in range(dataset_config[0].chunk_size):
        data_dict = {}
        data_dict[HandEyeDVRKConfig.camera_l.value[0]] = (
            np.ones((480, 640, 3), dtype=np.uint8) + i
        )
        data_dict[HandEyeDVRKConfig.camera_r.value[0]] = (
            np.ones((480, 640, 3), dtype=np.uint8) + 2 * i
        )

        data_container.add_data(data_dict)

    with h5_writer as writer:
        writer.write_chunk(data_container)


if __name__ == "__main__":
    test1()
