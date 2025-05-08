import logging
import os

from huggingface_hub import snapshot_download

from matgraphdb.core import MatGraphDB
from matgraphdb.core.edges import *
from matgraphdb.core.nodes import *
from matgraphdb.utils.config import config

MPNEARHULL_PATH = os.path.join(config.data_dir, "datasets", "MPNearHull")

# TODO: Need to find a way to deal with the case when the generator already exists

logger = logging.getLogger(__name__)


class MPNearHull(MatGraphDB):
    energy_above_hull_min = 0
    energy_above_hull_max = 0.2
    nsites_max = 100

    repo_id = "lllangWV/MPNearHull"
    repo_type = "dataset"

    def __init__(
        self,
        storage_path: str = MPNEARHULL_PATH,
        download: bool = True,
        from_scratch: bool = False,
        initialize_from_scratch: bool = True,
    ):

        if from_scratch:
            shutil.rmtree(storage_path)

        if download and not os.path.exists(storage_path):
            logger.info(f"Downloading dataset from {self.repo_id}")
            snapshot_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                local_dir=storage_path,
            )

        super().__init__(storage_path=storage_path)

        n_edge_generators = len(self.edge_generator_store.generator_names)
        n_node_generators = len(self.node_generator_store.generator_names)

        logger.debug(f"n_edge_generators: {n_edge_generators}")
        logger.debug(f"n_node_generators: {n_node_generators}")
        if initialize_from_scratch and (
            n_edge_generators == 0 and n_node_generators == 0
        ):
            self.initialize_nodes()
            self.initialize_edges()

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
