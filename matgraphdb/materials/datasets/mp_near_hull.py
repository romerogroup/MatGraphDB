import os
import shutil
import zipfile

import gdown

from matgraphdb.materials import MatGraphDB
from matgraphdb.materials.edges import *
from matgraphdb.materials.nodes import *
from matgraphdb.utils.config import config

MP_MATERIALS_PATH = os.path.join(config.data_dir, "raw", "MPNearHull", "materials")
MPNEARHULL_PATH = os.path.join(config.data_dir, "datasets", "MPNearHull")
DATASET_URL = "https://drive.google.com/uc?id=1zSmEQbV8pNvjWdhFuCwOeoOzvfoS5XKP"
RAW_DATASET_URL = "https://drive.google.com/uc?id=14guJqEK242XgRGEZA-zIrWyg4b-gX5zk"
RAW_DATASET_ZIP = os.path.join(config.data_dir, "raw", "MPNearHull_v0.0.1_raw.zip")
DATASET_ZIP = os.path.join(config.data_dir, "datasets", "MPNearHull_v0.0.1.zip")


class MPNearHull(MatGraphDB):
    energy_above_hull_min = 0
    energy_above_hull_max = 0.2
    nsites_max = 100

    def __init__(
        self,
        storage_path: str = MPNEARHULL_PATH,
        download: bool = True,
        from_raw_files=False,
    ):
        # Download dataset if it doesn't exist and download is True
        if download and not os.path.exists(storage_path) and not from_raw_files:
            print("Downloading dataset...")
            self.download_dataset()
            materials_store = None
        if from_raw_files and not download:
            print("Downloading raw materials data...")
            # Download raw data if it doesn't exist
            if not os.path.exists(MP_MATERIALS_PATH):
                print("Downloading raw materials data...")
                gdown.download(DATASET_URL, output=RAW_DATASET_ZIP, quiet=False)

                # Extract the raw dataset
                print("Extracting raw materials data...")
                with zipfile.ZipFile(RAW_DATASET_ZIP, "r") as zip_ref:
                    zip_ref.extractall(os.path.dirname(MP_MATERIALS_PATH))

                # Clean up the zip file
                os.remove(RAW_DATASET_ZIP)
                print("Raw materials data ready!")

            materials_store = MaterialStore(storage_path=MP_MATERIALS_PATH)
            super().__init__(storage_path=storage_path, materials_store=materials_store)

            self.initialize_nodes()
            self.initialize_edges()

        if not from_raw_files:
            super().__init__(storage_path=storage_path)

    @staticmethod
    def download_dataset():
        """Download and extract the MPNearHull dataset."""
        os.makedirs(os.path.dirname(DATASET_ZIP), exist_ok=True)

        # Download the dataset
        print("Downloading MPNearHull dataset...")
        gdown.download(DATASET_URL, output=DATASET_ZIP, quiet=False)

        # Extract the dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(MPNEARHULL_PATH))

        # Clean up the zip file
        os.remove(DATASET_ZIP)
        print("Dataset ready!")

    def initialize_nodes(self):

        node_generators = [
            {"generator_func": element},
            {"generator_func": chemenv},
            {"generator_func": crystal_system},
            {"generator_func": magnetic_state},
            {"generator_func": oxidation_state},
            {"generator_func": space_group},
            {"generator_func": wyckoff},
            {
                "generator_func": material_site,
                "generator_args": {"material_store": self.node_stores["material"]},
            },
            {
                "generator_func": material_lattice,
                "generator_args": {"material_store": self.node_stores["material"]},
            },
        ]

        for generator in node_generators:
            generator_func = generator.get("generator_func")
            generator_args = generator.get("generator_args", None)
            generator_name = generator_func.__name__
            self.add_node_generator(
                generator_func=generator_func,
                generator_args=generator_args,
            )

    def initialize_edges(self):
        edge_generators = [
            {
                "generator_func": element_element_neighborsByGroupPeriod,
                "generator_args": {"element_store": self.node_stores["element"]},
            },
            {
                "generator_func": element_oxiState_canOccur,
                "generator_args": {
                    "element_store": self.node_stores["element"],
                    "oxiState_store": self.node_stores["oxidation_state"],
                },
            },
            {
                "generator_func": material_chemenv_containsSite,
                "generator_args": {
                    "material_store": self.node_stores["material"],
                    "chemenv_store": self.node_stores["chemenv"],
                },
            },
            {
                "generator_func": material_crystalSystem_has,
                "generator_args": {
                    "material_store": self.node_stores["material"],
                    "crystal_system_store": self.node_stores["crystal_system"],
                },
            },
            {
                "generator_func": material_element_has,
                "generator_args": {
                    "material_store": self.node_stores["material"],
                    "element_store": self.node_stores["element"],
                },
            },
            {
                "generator_func": material_lattice_has,
                "generator_args": {
                    "material_store": self.node_stores["material"],
                    "lattice_store": self.node_stores["material_lattice"],
                },
            },
            {
                "generator_func": material_spg_has,
                "generator_args": {
                    "material_store": self.node_stores["material"],
                    "spg_store": self.node_stores["space_group"],
                },
            },
            {
                "generator_func": element_chemenv_canOccur,
                "generator_args": {
                    "element_store": self.node_stores["element"],
                    "chemenv_store": self.node_stores["chemenv"],
                    "material_store": self.node_stores["material"],
                },
            },
        ]

        for generator in edge_generators[:]:
            generator_func = generator.get("generator_func")
            generator_args = generator.get("generator_args", None)
            self.add_edge_generator(
                generator_func=generator_func,
                generator_args=generator_args,
                run_immediately=True,
            )
