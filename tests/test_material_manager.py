import logging
import unittest
import shutil
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from pymatgen.core import Structure

from matgraphdb.data.material_manager import MaterialDatabaseManager, convert_coordinates, check_all_params_provided, perform_symmetry_analysis

# logger=logging.getLogger('matgraphdb')
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)


class TestMaterialDatabaseManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = MaterialDatabaseManager(db_dir=self.temp_dir, n_cores=2)

    def tearDown(self):
        # Remove the temporary directory after the test
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_add_structure(self):
        # Test adding a material with structure, species, and lattice
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        species = ["Fe", "O"]
        lattice = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        material = dict(coords=coords, species=species, lattice=lattice, properties={"density": 1.0})

        # Add a material and check if the result is a valid entry
        entry_data = self.manager.add(**material)
        self.assertIn('formula', entry_data)
        self.assertIn('density', entry_data)
        self.assertEqual(entry_data['density'], 1.0)

        # Ensure the material is added to the database
        table = self.manager.read()
        df = table.to_pandas()

        self.assertEqual(df.shape[0], 1)
        self.assertIn('formula', df.columns)

    def test_add_many_materials(self):
        # Create multiple materials and test bulk addition
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        species = ["Fe", "O"]
        lattice = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        material = dict(coords=coords, species=species, lattice=lattice, properties={"density": 1.0})
        materials = [material for _ in range(10)]

        # Add materials and verify they were added successfully
        self.manager.add_many(materials)
        table = self.manager.read()
        df = table.to_pandas()

        self.assertEqual(df.shape[0], 10)
        self.assertIn('density', df.columns)

    def test_update_material(self):
        # Add a material to update later
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        species = ["Fe", "O"]
        lattice = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        material = dict(coords=coords, species=species, lattice=lattice, properties={"density": 1.0})
        entry_data = self.manager.add(**material)

        # Modify the material and update it
        updated_data = entry_data.copy()
        updated_data['id'] = 0
        updated_data['density'] = 2.0
        self.manager.update([updated_data])

        # Verify the update
        table = self.manager.read()
        df = table.to_pandas()

        self.assertEqual(df.iloc[0]['density'], 2.0)

    def test_delete_material(self):
        # Add a material to be deleted
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        species = ["Fe", "O"]
        lattice = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        material = dict(coords=coords, species=species, lattice=lattice, properties={"density": 1.0})
        self.manager.add(**material)

        # Delete the material
        ids = [0]
        self.manager.delete(ids)

        # Verify deletion
        table = self.manager.read()
        df = table.to_pandas()

        self.assertEqual(df.shape[0], 0)

if __name__ == '__main__':
    unittest.main()
