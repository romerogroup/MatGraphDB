
___

# 0.3.0 (02-08-2025)

##### Bugs
- None identified

##### New Features
- Implemented heterogeneous graph encoder training infrastructure
- Added PyTorch Geometric (PyG) infrastructure for graph machine learning
- Included new periodic table values and coordination geometries to support advanced material analysis
- Added new dataset initialization in the materials.datasets module to support the new MPNearHull functionality

##### Documentation
- Updated _version.py and CHANGELOG.md due to new release

##### Maintenance
- Streamlined dependency management in pyproject.toml
- Improved initialization logic to handle dataset presence and downloading options
- Updated .gitignore to exclude the /data/external directory, streamlining data management and ensuring that only relevant data directories are tracked
- Added .gitattributes to track data
- Added comprehensive tests for the new GraphBuilder functionality and updated existing tests to reflect changes in edge data structure

___

___

# 0.2.0 (01-13-2025)

##### Bugs
- None identified
##### New features
- None identified
##### Documentation updates
- None identified
##### Maintenance
- Improved logging and error handling across various modules for better traceability during operations
- Updated _version.py and CHANGELOG.md due to new release

___

___

# 0.1.0 (01-11-2025)

##### Bugs
- Fixed bugs in MatDB and add tests for MatDB
- Changes due to the API change in MatDB and changes to ParquetDB
- bug: typo in env.tml

##### New Features
- Added CalculationManager and core utilities for material calculations
- Add MatGraphDB class for advanced material analysis and management
- Add MaterialStore class for comprehensive material management
- Add custom PyArrow classes for enhanced material structure handling
- Added method to get element_properties
- Added better docs to coord_geometry module
- Added script to handle loading of coordination data previously stored in JSON, now in Parquet format

##### Documentation
- Update README.md to streamline the introduction and improve clarity
- Refactor Nodes class documentation in nodes.py
- Updated to have matgraphdb to have __version__ variable
- Update _version.py and CHANGELOG.md due to new release

##### Maintenance
- Remove deprecated files and scripts from the sandbox and old_scripts directories
- Improved logging and error handling across various modules
- Refactor edge and node management in MatGraphDB
- Refactor `NodeStore` usage across multiple modules
- Removed old test data
- Cleaned up whitespace in the SpaceGroupNodes class
- Refactor GraphStore methods for improved clarity and consistency
- Refactor GraphStore initialization and enhance directory structure
- Refactor MaterialStore for improved data management and functionality
- Refactor EdgeStore and GraphStore initialization logic
- Update .gitignore, removed examples
- Moved old sandbox directories to old_scripts
- Removed old chem util resource formats
- Updated environmental configuration management 
- Updated directory structure for examples
- Updated publish workflow
- Updated dependencies
- Updated GitHub workflow scripts
- Enhanced configuration and logging settings in MatGraphDB

___

___

# 0.0.3 (10-03-2024)

##### Bugs
- None identified
##### New Features
- Added new python deployment workflow. On release, this will build and deploy on PyPI, generate a changelog from the commits, update the repo version, and update the release notes with the current change version changelog
##### Documentation updates
- None identified
##### Maintenance
- Moved old scripts to sandbox
- Removed unused file

___
