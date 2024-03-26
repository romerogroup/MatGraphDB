
```bash
.\bin\neo4j-admin.bat database import full --nodes import\elements.csv import\materials.csv import\chemenv_names.csv import\crystal_systems.csv import\chemenv_element_names.csv --relationships import\chemenv_chemenv_geometric-electric.csv import\chemenvElement_chemenvElement_geometric-electric.csv import\element_element_geometric-electric.csv import\materials_chemenvElement.csv import\materials_elements.csv import\materials_chemenv.csv import\chemenv_elements.csv --overwrite-destination test
```


```bash
.\bin\neo4j-admin.bat database import full --nodes import\elements.csv --relationships import\Element_Element.csv --overwrite-destination test
```