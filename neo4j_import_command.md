
```bash
.\bin\neo4j-admin.bat database import full --nodes import\elements.csv import\materials.csv import\chemenv_names.csv import\crystal_systems.csv import\chemenv_element_names.csv import\spg.csv import\magnetic_states.csv import\oxidation_states.csv --relationships import\chemenv_chemenv_geometric-electric.csv import\chemenvElement_chemenvElement_geometric-electric.csv import\element_element_geometric-electric.csv import\materials_chemenvElement.csv import\materials_elements.csv import\materials_chemenv.csv import\chemenv_elements.csv --overwrite-destination matgraphdb
```

```bash
.\bin\neo4j-admin.bat database import full --nodes import\elements.csv import\materials.csv import\chemenv_names.csv import\crystal_systems.csv import\chemenv_element_names.csv import\spg.csv import\magnetic_states.csv import\oxidation_states.csv --relationships import\chemenv_chemenv_geometric-electric.csv import\chemenvElement_chemenvElement_geometric-electric.csv import\element_element_geometric-electric.csv import\materials_chemenvElement.csv import\materials_elements.csv import\materials_chemenv.csv import\chemenv_elements.csv --overwrite-destination test_import
```

```bash
.\bin\neo4j-admin.bat database import full --nodes import\elements.csv import\materials.csv import\chemenv_names.csv import\crystal_systems.csv import\chemenv_element_names.csv import\spg.csv import\magnetic_states.csv import\oxidation_states.csv --relationships import\chemenvElement_chemenvElement_geometric-electric.csv import\chemenv_chemenv_geometric-electric.csv import\element_element_geometric-electric.csv import\chemenv_elements.csv import\materials_chemenvElement.csv import\materials_elements.csv import\materials_chemenv.csv import\materials_crystal_system.csv import\materials_spg.csv --overwrite-destination matgraphdb
```


```bash
.\bin\neo4j-admin.bat database import full --nodes import\elements.csv --relationships import\Element_Element.csv --overwrite-destination test
```



```cypher
CALL gds.graph.project(
  'materials_chemenvElements',
  ['ChemenvElement','Material'],
    {
                    GEOMETRIC_ELECTRIC_CONNETS: {
                    orientation: 'UNDIRECTED',
                    properties: 'weight'
                    },
                    COMPOSED_OF: {
                        orientation: 'UNDIRECTED',
                        properties'weight'
                    }
                }
)


CALL gds.fastRP.stream.estimate('materials_chemenvElements', {embeddingDimension: 128})
YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory


CALL gds.fastRP.stream('materials_chemenvElements',
  {
    embeddingDimension: 128,
    randomSeed: 42
  }
)
YIELD nodeId, embedding
```