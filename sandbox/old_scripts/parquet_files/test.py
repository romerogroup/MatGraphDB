import pandas as pd

import pyarrow as pa
from pyarrow import json
import pyarrow.parquet as pq
import pyarrow.compute as pc

import os

current_dir='examples/parquet_files'


# 
json_dir='data/production/materials_project/json_database'
# json_dir='data/raw/materials_project_nelements_2/json_database'
json_file_1=os.path.join(json_dir,'mp-1000.json')
json_file_2=os.path.join(json_dir,'mp-1001.json')
# Example DataFrame creation
data = {
    'floats': [1.0, 2.0, 3.0],
    'ints': [1, 2, 3],
    'array_floats': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    'array_strings': [['a',
'b'], ['c',
'd'], ['e',
'f']],
    'array_ints': [[1, 2], [3, 4], [5, 6]],
    'dicts': [{'key1': 10, 'key2': 'value1'}, {'key1': 20, 'key2': 'value2'}, {'key1': 30, 'key2': 'value3'}],
    'strings': ['foo',
'bar',
'baz']
}

# Convert DataFrame to PyArrow Table, specifying how to handle arrays
# table_dict = pa.Table.from_pandas(df, preserve_index=False)
# print(table)

# print('-'*100)
# Convert DataFrame to PyArrow Table

t_string=pa.string()
t_int=pa.int64()
t_float=pa.float64()
t_bool=pa.bool_()
table_schema = pa.schema([('material_id', t_string),
                       ('nsites', t_int),
                       ('elements', pa.list_(t_string)),
                       ('nelements', t_int),
                       ('formula_pretty', t_string),
                       ('volume', t_int),
                       ('density', t_int),
                       ('density_atomic', t_int),
                       ('symmetry', pa.struct([('crystal_system', t_string),
                                               ('symbol', t_string),
                                               ('number', t_string),
                                               ('point_group', t_string),
                                               ('symprec', t_float),
                                               ('space_group', t_int)])),
                       ('structure', pa.struct([('@module', t_string),
                                                ('@class', t_string),
                                                ('charge', t_int),
                                                ('lattice', pa.struct([
                                                                    ('matrix', pa.list_(pa.list_(t_float))),
                                                                    ('pbc', pa.list_(t_bool)),
                                                                    ('a', t_float),
                                                                    ('b', t_float),
                                                                    ('c', t_float),
                                                                    ('alpha', t_float),
                                                                    ('beta', t_float),
                                                                    ('gamma', t_float),
                                                                    ('volumne', t_float)])),
                                                 ('sites', pa.list_(pa.struct([
                                                                    ('species', pa.list_(pa.struct([
                                                                                            ('element', t_string),
                                                                                            ('occu', t_float)]))),
                                                                    ('abc', pa.list_(t_float)),
                                                                    ('xyz', pa.list_(t_float)),
                                                                    ('properties', pa.struct([('magmom', t_float)]))])))
                                                ])),
                        ('energy_per_atom', t_float),
                        ('formation_energy_per_atom', t_float),
                        ('energy_above_hull', t_float),
                        ('is_stable', t_bool),
                        ('band_gap', t_float),
                        ('cbm', t_float),
                        ('vbm', t_float),
                        ('efermi', t_float),
                        ('is_gap_direct', t_bool),
                        ('is_metal', t_bool),
                        ('is_magnetic', t_bool),
                        ('ordering', t_string),
                        ('total_magnetization', t_float),
                        ('total_magnetization_normalized_vol', t_float),
                        ('num_magnetic_sites', t_int),
                        ('num_unique_magnetic_sites', t_int),
                        ('k_voigt', t_float),
                        ('k_reuss', t_float),
                        ('k_vrh', t_float),
                        ('g_voigt', t_float),
                        ('g_reuss', t_float),
                        ('g_vrh', t_float),
                        ('universal_anisotropy', t_float),
                        ('homogeneous_poisson', t_float),
                        ('e_total', t_float),
                        ('e_ionic', t_float),
                        ('e_electronic', t_float),
                        ('coordination_environments_multi_weight', pa.list_(pa.struct([('ce_symbol', t_string),
                                                                                            ('ce_fraction', t_float),
                                                                                            ('csm', t_float),
                                                                                            ('permutation', pa.list_(t_int))]))),
                        ('coordination_multi_connections', pa.list_(pa.list_(t_int))),
                        ('coordination_multi_numbers', pa.list_(t_int)),
                        ('chargemol_bonding_connections', pa.list_(pa.list_(t_int))),
                        ('chargemol_bonding_orders', pa.list_(t_float)),
                        ('chargemol_net_atomic_charges', pa.list_(t_float)),
                        ('chargemol_squared_moments', pa.list_(t_float)),
                        ('chargemol_cubed_moments', pa.list_(t_float)),
                        ('chargemol_fourth_moments', pa.list_(t_float)),
                        ('geometric_consistent_bond_connections', pa.list_(pa.list_(t_int))),
                        ('geometric_consistent_bond_orders', pa.list_(t_float)),
                        ('electric_consistent_bond_connections', pa.list_(pa.list_(t_int))),
                        ('electric_consistent_bond_orders', pa.list_(t_float)),
                        ('geometric_electric_consistent_bond_connections', pa.list_(pa.list_(t_int))),
                        ('geometric_electric_consistent_bond_orders', pa.list_(t_float)),
                        ('bond_cutoff_connections', pa.list_(pa.list_(t_int))),
                        ('wyckoffs', pa.list_(t_string)),
                        ('oxidation_states', pa.struct([('possible_species', pa.list_(t_string)),
                                                        ('possible_valences', pa.list_(t_int)),
                                                        ('method', t_string)])),
                        ('last_updated', t_string),
                        ('uncorrected_energy_per_atom', t_float),
                        ('equilibrium_reaction_energy_per_atom', t_float),
                        ('n', t_float),
                        ('e_ij_max', t_float),
                        ('weighted_surface_energy_EV_PER_ANG2', t_float),
                        ('weighted_surface_energy', t_float),
                        ('weighted_work_function', t_float),
                        ('surface_anisotropy', t_float),
                        ('shape_factor', t_float),
                        ('has_reconstructed', t_bool),
                        ('possible_species', pa.list_(t_string)),
                        ('has_props', pa.struct([
                                    ('materials', t_bool),
                                    ('thermo', t_bool),
                                    ('xas', t_bool),
                                    ('grain_boundaries', t_bool),
                                    ('chemenv',t_bool),
                                    ('electronc_structure', t_bool),
                                    ('absorption',t_bool),
                                    ('dos',t_bool),
                                    ('magnetism',t_bool),
                                    ('elasticity',t_bool),
                                    ('dielectric',t_bool),
                                    ('piezoelectric',t_bool),
                                    ('surface_properties',t_bool),
                                    ('oxi_states',t_bool),
                                    ('provenance',t_bool),
                                    ('charge_density',t_bool),
                                    ('eos',t_bool),
                                    ('phonons',t_bool),
                                    ('insertion_electrodes',t_bool),
                                    ('substrates',t_bool)])),
                        ('theoretical', t_bool),
                        ('feature_vectors', pa.struct([
                                                    ('sine_coulomb_matrix', pa.struct([('values', pa.list_(pa.list_(t_float))),
                                                                                        ('feature_names', pa.list_(t_string))])),
                                                    ('element_fraction', pa.struct([('values', pa.list_(pa.list_(t_float))),
                                                                                        ('feature_names', pa.list_(t_string))])),
                                                    ('element_property', pa.struct([('values', pa.list_(pa.list_(t_float))),
                                                                                        ('feature_names', pa.list_(t_string))])),
                                                    ('xrd_pattern', pa.struct([('values', pa.list_(pa.list_(t_float))),
                                                                                        ('feature_names', pa.list_(t_string))])),
                                                                                            ]))
])
                        
# parse_options = json.ParseOptions(allow_decimal=True)
table_1=json.read_json(json_file_1)
table_1=table_1.drop(['composition',
'composition_reduced'])
table_2=json.read_json(json_file_2)
table_2=table_2.drop(['composition',
'composition_reduced'])
table=pa.concat_tables([table_1,table_2],
                       promote_options="permissive")

# chunk_array=table['structure']


# lattice=chunk_array.chunk(0)
# print(type(table['structure']))

# lattice=pc.struct_field(table['structure'], 'lattice')
# lattice_matrix=pc.struct_field(lattice, 'matrix')
# print(lattice_matrix)

table=pa.Table.from_pydict(data)

# print(table)

# Save as Parquet file
# pq.write_table(table, os.path.join(current_dir, 'test_pydict.parquet')   )



# read_df = pq.read_table(os.path.join(current_dir, 'test_pydict.parquet') ).to_pandas()


# print(read_df.head())




def convert_values_into_array(data,default=None):
    for key, value in data.items():
        print(key)
        data[key] = [data.get(key, default)]
    return data


def get_data(key, default=None):
    return [data.get(key, default)]

# table=pa.Table.from_pydict(data,schema=table_schema)
# table = pa.Table.from_pandas(df)
# table=pa.Table.from_pydict(data,schema=table_schema)

# arrays=[]
# names=list(data.keys())
# for key in data.keys():
#     arrays.append(get_data(key))
converted_data=convert_values_into_array(data)

print(converted_data)
# table=pa.Table.from_arrays(arrays,names)
# table=pa.Table.from_pydict(converted_data,schema=table_schema)
table=pa.Table.from_pydict(converted_data)

# print(converted_data)
# table=pa.Table.from_pydict(converted_data,schema=table_schema)
# table=pa.Table.from_pydict(converted_data,schema=table_schema)

print(table)