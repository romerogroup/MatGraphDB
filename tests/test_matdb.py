import logging
import unittest
import shutil
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from pymatgen.core import Structure

from matgraphdb.data.matdb import MatDB, convert_coordinates, check_all_params_provided, perform_symmetry_analysis

class TestMaterialDatabaseManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.matdb = MatDB(db_path=self.temp_dir, n_cores=2)

    def tearDown(self):
        # Remove the temporary directory after the test
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_add_structure(self):
        # Test adding a material with structure, species, and lattice
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        species = ["Fe", "O"]
        lattice = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        material = dict(coords=coords, species=species, lattice=lattice, properties={'electronic_structure':{"band_gap": 1.0}})

        # Add a material and check if the result is a valid entry
        self.matdb.add(**material)
        
        
        table = self.matdb.read()
        
        self.assertIn('core.formula', table.column_names)
        self.assertIn('electronic_structure.band_gap', table.column_names)
        self.assertEqual(table['electronic_structure.band_gap'].combine_chunks().to_pylist()[0], 1.0)

        # Ensure the material is added to the database
        table = self.matdb.read()
        df = table.to_pandas()

        self.assertEqual(df.shape[0], 1)
        self.assertIn('core.formula', df.columns)
        self.assertIn('electronic_structure.band_gap', df.columns)
        self.assertEqual(df.iloc[0]['electronic_structure.band_gap'], 1.0)

    def test_add_many_materials(self):
        # Create multiple materials and test bulk addition
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        species = ["Fe", "O"]
        lattice = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        material = dict(coords=coords, species=species, lattice=lattice, properties={'electronic_structure':{"band_gap": 1.0}})
        materials = [material for _ in range(10)]

        # Add materials and verify they were added successfully
        self.matdb.add_many(materials)
        table = self.matdb.read()
        df = table.to_pandas()

        self.assertEqual(df.shape[0], 10)
        self.assertIn('electronic_structure.band_gap', df.columns)

    def test_update_material(self):
        # Add a material to update later
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        species = ["Fe", "O"]
        lattice = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        material = dict(coords=coords, species=species, lattice=lattice, properties={'electronic_structure':{"band_gap": 1.0}})
        self.matdb.add(**material)
        
        table = self.matdb.read()
        df = table.to_pandas()
        self.assertEqual(df.shape[0], 1)
        self.assertIn('electronic_structure.band_gap', df.columns)
        self.assertEqual(df.iloc[0]['electronic_structure.band_gap'], 1.0)

        # Modify the material and update it
        update_dict={
            'id':0,
            'electronic_structure':{
                'band_gap':2.0
            },
            'core':{
                'lattice':[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
            }
        }
        self.matdb.update([update_dict])

        # Verify the update
        table = self.matdb.read()
        df = table.to_pandas()

        self.assertEqual(df.iloc[0]['electronic_structure.band_gap'], 2.0)
        assert np.array_equal(table['core.lattice'].combine_chunks().to_numpy_ndarray()[0], np.array(update_dict['core']['lattice']))

    def test_delete_material(self):
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        species = ["Fe", "O"]
        lattice = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        material = dict(coords=coords, species=species, lattice=lattice, properties={'electronic_structure':{"band_gap": 1.0}})
        materials = [material for _ in range(10)]

        self.matdb.add_many(materials)

        # Delete the material
        ids = [0]
        self.matdb.delete(ids=ids)

        # Verify deletion
        table = self.matdb.read()
        df = table.to_pandas()
        self.assertEqual(df.shape[0], 9)

if __name__ == '__main__':
    print('testing matdb')
    
    # unittest.TextTestRunner().run(TestMaterialDatabaseManager('test_add_many_materials'))
    # unittest.TextTestRunner().run(TestMaterialDatabaseManager('test_add_many_materials'))
    # unittest.TextTestRunner().run(TestMaterialDatabaseManager('test_update_material'))
    # unittest.TextTestRunner().run(TestMaterialDatabaseManager('test_delete_material'))
    unittest.main()
