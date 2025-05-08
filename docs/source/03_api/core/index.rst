.. _materials-api-index:

Materials API
===================================

The Materials API provides the fundamental functionality of MatGraphDB, offering a robust interface for managing a graph database. This module contains the essential classes and methods that enable database-like operations 
The core components include:

- :class:`MatGraphDB <matgraphdb.materials.core.MatGraphDB>` - The main interface class that provides database-like operations over Parquet files. This class handles data storage, retrieval, querying, schema evolution, and complex data type management through an intuitive API that wraps PyArrow's functionality.

- :class:`MaterialStore <matgraphdb.materials.nodes.materials.MaterialStore>` - A store for managing materials in a graph database.


Node Generators
========================

- :func:`element <matgraphdb.core.nodes.generators.element>` - A function that generates the elements of a material.

- :func:`chemenv <matgraphdb.core.nodes.generators.chemenv>` - A function that generates the chemical environments of a material.

- :func:`crystal_system <matgraphdb.core.nodes.generators.crystal_system>` - A function that generates the crystal systems of a material.

- :func:`magnetic_state <matgraphdb.core.nodes.generators.magnetic_state>` - A function that generates the magnetic states of a material.

- :func:`oxidation_state <matgraphdb.core.nodes.generators.oxidation_state>` - A function that generates the oxidation states of a material.

- :func:`space_group <matgraphdb.core.nodes.generators.space_group>` - A function that generates the space groups of a material.

- :func:`wyckoff <matgraphdb.core.nodes.generators.wyckoff>` - A function that generates the wyckoffs of a material.


Edge Generators
========================

- :func:`element_element_neighborsByGroupPeriod <matgraphdb.core.edges.element_element_neighborsByGroupPeriod>` - A function that generates the neighbors of an element by group and period.

- :func:`element_element_bonds <matgraphdb.core.edges.element_element_bonds>` - A function that generates the bonds of an element.

- :func:`element_oxiState_canOccur <matgraphdb.core.edges.element_oxiState_canOccur>` - A function that generates the possible oxidation states of an element.

- :func:`material_chemenv_containsSite <matgraphdb.core.edges.material_chemenv_containsSite>` - A function that generates the sites of a material.

- :func:`material_crystalSystem_has <matgraphdb.core.edges.material_crystalSystem_has>` - A function that generates the crystal system of a material.

- :func:`material_element_has <matgraphdb.core.edges.material_element_has>` - A function that generates the elements of a material.

- :func:`material_lattice_has <matgraphdb.core.edges.material_lattice_has>` - A function that generates the lattice of a material.

- :func:`material_spg_has <matgraphdb.core.edges.material_spg_has>` - A function that generates the space group of a material.

- :func:`element_chemenv_canOccur <matgraphdb.core.edges.element_chemenv_canOccur>` - A function that generates the possible oxidation states of an element.

- :func:`spg_crystalSystem_isApart <matgraphdb.core.edges.spg_crystalSystem_isApart>` - A function that generates the crystal system of a material.


.. toctree::
   :maxdepth: 2
   
   matgraphdb_base
   edge_generators
   node_generators
   material_store
