import logging
import os
import warnings
from glob import glob

import numpy as np
import pandas as pd

from matgraphdb.core import NodeStore
from matgraphdb.materials.nodes.materials import MaterialNodes

logger = logging.getLogger(__name__)


class MaterialLatticeNodes(NodeStore):
    def __init__(self, storage_path: str, material_nodes_path="data/nodes/materials"):
        super().__init__(
            storage_path=storage_path,
            initialize_kwargs={"material_nodes_path": material_nodes_path},
        )

    def initialize(self, material_nodes_path="data/nodes/materials"):
        """
        Creates Lattice nodes if no file exists, otherwise loads them from a file.
        """
        self.name_column = "material_id"
        # Retrieve material nodes with lattice properties
        try:
            material_nodes = MaterialNodes(material_nodes_path)

            table = material_nodes.read(
                columns=[
                    "structure.lattice.a",
                    "structure.lattice.b",
                    "structure.lattice.c",
                    "structure.lattice.alpha",
                    "structure.lattice.beta",
                    "structure.lattice.gamma",
                    "structure.lattice.volume",
                    "structure.lattice.pbc",
                    "structure.lattice.matrix",
                    "id",
                    "core.material_id",
                ]
            )

            for i, column in enumerate(table.columns):
                field = table.schema.field(i)
                field_name = field.name
                if "." in field_name:
                    field_name = field_name.split(".")[-1]
                if "id" == field_name:
                    field_name = "material_node_id"
                new_field = field.with_name(field_name)
                table = table.set_column(i, new_field, column)

        except Exception as e:
            logger.error(f"Error creating lattice nodes: {e}")
            return None

        return table
