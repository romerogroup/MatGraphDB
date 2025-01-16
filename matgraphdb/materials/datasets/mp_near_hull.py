import os

from matgraphdb.materials import MatGraphDB
from matgraphdb.materials.edges import *
from matgraphdb.materials.nodes import *
from matgraphdb.utils.config import config

MP_MATERIALS_PATH = os.path.join(config.data_dir, "raw", "MPNearHull", "materials")

MPNEARHULL_PATH = os.path.join(config.data_dir, "datasets", "MPNearHull")


class MPNearHull(MatGraphDB):
    energy_above_hull_min = 0
    energy_above_hull_max = 0.2
    nsites_max = 100

    def __init__(self, storage_path: str = MPNEARHULL_PATH):
        # Initialize with your custom store
        materials_store = MaterialStore(storage_path=MP_MATERIALS_PATH)

        super().__init__(storage_path=storage_path, materials_store=materials_store)

        self.initialize_nodes()
        self.initialize_edges()

    def initialize_nodes(self):

        node_generators = [
            {"generator_func": elements},
            {"generator_func": chemenvs},
            {"generator_func": crystal_systems},
            {"generator_func": magnetic_states},
            {"generator_func": oxidation_states},
            {"generator_func": space_groups},
            {"generator_func": wyckoffs},
            {
                "generator_func": material_sites,
                "generator_args": {"material_store": self.node_stores["materials"]},
            },
            {
                "generator_func": material_lattices,
                "generator_args": {"material_store": self.node_stores["materials"]},
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
                "generator_args": {"element_store": self.node_stores["elements"]},
            },
            {
                "generator_func": element_oxiState_canOccur,
                "generator_args": {
                    "element_store": self.node_stores["elements"],
                    "oxiState_store": self.node_stores["oxidation_states"],
                },
            },
            {
                "generator_func": material_chemenv_containsSite,
                "generator_args": {
                    "material_store": self.node_stores["materials"],
                    "chemenv_store": self.node_stores["chemenvs"],
                },
            },
            {
                "generator_func": material_crystalSystem_has,
                "generator_args": {
                    "material_store": self.node_stores["materials"],
                    "crystal_system_store": self.node_stores["crystal_systems"],
                },
            },
            {
                "generator_func": material_element_has,
                "generator_args": {
                    "material_store": self.node_stores["materials"],
                    "element_store": self.node_stores["elements"],
                },
            },
            {
                "generator_func": material_lattice_has,
                "generator_args": {
                    "material_store": self.node_stores["materials"],
                    "lattice_store": self.node_stores["material_lattices"],
                },
            },
            {
                "generator_func": material_spg_has,
                "generator_args": {
                    "material_store": self.node_stores["materials"],
                    "spg_store": self.node_stores["space_groups"],
                },
            },
            {
                "generator_func": element_chemenv_canOccur,
                "generator_args": {
                    "element_store": self.node_stores["elements"],
                    "chemenv_store": self.node_stores["chemenvs"],
                    "material_store": self.node_stores["materials"],
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


if __name__ == "__main__":
    if os.path.exists(MPNEARHULL_PATH):
        shutil.rmtree(MPNEARHULL_PATH)
    mdb = MPNearHull(storage_path=MPNEARHULL_PATH)
