from enum import Enum

import pyarrow as pa
from matgraphdb.data.utils import material_property_schema_list

class NodeTypes(Enum):
    ELEMENT='ELEMENT'
    PRE_IMPUTED_ELEMENT='PRE_IMPUTED_ELEMENT'
    CHEMENV='CHEMENV'
    CRYSTAL_SYSTEM='CRYSTAL_SYSTEM'
    MAGNETIC_STATE='MAGNETIC_STATE'
    SPACE_GROUP='SPACE_GROUP'
    OXIDATION_STATE='OXIDATION_STATE'
    MATERIAL='MATERIAL'
    SPG_WYCKOFF='SPG_WYCKOFF'
    CHEMENV_ELEMENT='CHEMENV_ELEMENT'
    LATTICE='LATTICE'
    SITE='SITE'


class RelationshipTypes(Enum):

    MATERIAL_SPG=f'{NodeTypes.MATERIAL.value}-HAS-{NodeTypes.SPACE_GROUP.value}'
    MATERIAL_CRYSTAL_SYSTEM=f'{NodeTypes.MATERIAL.value}-HAS-{NodeTypes.CRYSTAL_SYSTEM.value}'
    MATERIAL_LATTICE=f'{NodeTypes.MATERIAL.value}-HAS-{NodeTypes.LATTICE.value}'
    MATERIAL_SITE=f'{NodeTypes.MATERIAL.value}-HAS-{NodeTypes.SITE.value}'

    MATERIAL_CHEMENV=f'{NodeTypes.MATERIAL.value}-HAS-{NodeTypes.CHEMENV.value}'
    MATERIAL_CHEMENV_ELEMENT=f'{NodeTypes.MATERIAL.value}-HAS-{NodeTypes.CHEMENV_ELEMENT.value}'
    MATERIAL_ELEMENT=f'{NodeTypes.MATERIAL.value}-HAS-{NodeTypes.ELEMENT.value}'

    ELEMENT_OXIDATION_STATE=f'{NodeTypes.ELEMENT.value}-CAN_OCCUR-{NodeTypes.OXIDATION_STATE.value}'
    ELEMENT_CHEMENV=f'{NodeTypes.ELEMENT.value}-CAN_OCCUR-{NodeTypes.CHEMENV.value}'

    ELEMENT_GEOMETRIC_CONNECTS_ELEMENT=f'{NodeTypes.ELEMENT.value}-GEOMETRIC_CONNECTS-{NodeTypes.ELEMENT.value}'
    ELEMENT_ELECTRIC_CONNECTS_ELEMENT=f'{NodeTypes.ELEMENT.value}-ELECTRIC_CONNECTS-{NodeTypes.ELEMENT.value}'
    ELEMENT_GEOMETRIC_ELECTRIC_CONNECTS_ELEMENT=f'{NodeTypes.ELEMENT.value}-GEOMETRIC_ELECTRIC_CONNECTS-{NodeTypes.ELEMENT.value}'
    ELEMENT_GROUP_PERIOD_CONNECTS_ELEMENT=f'{NodeTypes.ELEMENT.value}-GROUP_PERIOD_CONNECTS-{NodeTypes.ELEMENT.value}'

    CHEMENV_GEOMETRIC_CONNECTS_CHEMENV=f'{NodeTypes.CHEMENV.value}-GEOMETRIC_CONNECTS-{NodeTypes.CHEMENV.value}'
    CHEMENV_ELECTRIC_CONNECTS_CHEMENV=f'{NodeTypes.CHEMENV.value}-ELECTRIC_CONNECTS-{NodeTypes.CHEMENV.value}'
    CHEMENV_GEOMETRIC_ELECTRIC_CONNECTS_CHEMENV=f'{NodeTypes.CHEMENV.value}-GEOMETRIC_ELECTRIC_CONNECTS-{NodeTypes.CHEMENV.value}'

t_string=pa.string()
t_int=pa.int64()
t_float=pa.float64()
t_bool=pa.bool_()


##############################################################################################################################
# Elements

# Create parquet schem from the column names above
element_property_schema_list = [
    pa.field('long_name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('symbol', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('abundance_universe', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('abundance_solar', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('abundance_meteor', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('abundance_crust', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('abundance_ocean', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('abundance_human', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('adiabatic_index', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('allotropes', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('appearance', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('atomic_mass', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('atomic_number', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('block', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('boiling_point', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('classifications_cas_number', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('classifications_cid_number', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('classifications_rtecs_number', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('classifications_dot_numbers', t_string, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('classifications_dot_hazard_class', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('conductivity_thermal', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('cpk_hex', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('critical_pressure', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('critical_temperature', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('crystal_structure', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('density_stp', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('discovered_year', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('discovered_by', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('discovered_location', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('electron_affinity', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('electron_configuration', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('electron_configuration_semantic', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('electronegativity_pauling', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('energy_levels', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('gas_phase', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('group', t_int, metadata={'encoder':'IntegerOneHotEncoder(dtype=torch.float32)'}),
    pa.field('extended_group', t_int, metadata={'encoder':'IntegerOneHotEncoder(dtype=torch.float32)'}),
    pa.field('half_life', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('heat_specific', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('heat_vaporization', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('heat_fusion', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('heat_molar', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('isotopes_known', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('isotopes_stable', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('isotopic_abundances', t_string, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('lattice_angles', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('lattice_constants', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('lifetime', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('magnetic_susceptibility_mass', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('magnetic_susceptibility_molar', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('magnetic_susceptibility_volume', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('oxidation_states', pa.list_(t_float), metadata={'encoder':'OxidationStatesEncoder(dtype=torch.float32)'}),
    pa.field('magnetic_type', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('melting_point', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('molar_volume', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('neutron_cross_section', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('neutron_mass_absorption', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('period', t_int, metadata={'encoder':'IntegerOneHotEncoder(dtype=torch.float32)'}),
    pa.field('phase', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('quantum_numbers', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('radius_calculated', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('radius_empirical', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('radius_covalent', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('radius_vanderwaals', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('refractive_index', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('series', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('source', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('space_group_name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('space_group_number', t_int, metadata={'encoder':'SpaceGroupOneHotEncoder(dtype=torch.float32)'}),
    pa.field('speed_of_sound', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('summary', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('valence_electrons', t_int, metadata={'encoder':'IntegerOneHotEncoder(dtype=torch.float32)'}),
    pa.field('conductivity_electric', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('electrical_resistivity', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('electrical_type', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('modulus_bulk', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('modulus_shear', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('modulus_young', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('poisson_ratio', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('coefficient_of_linear_thermal_expansion', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('hardness_vickers', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('hardness_brinell', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('hardness_mohs', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('superconduction_temperature', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('is_actinoid', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_alkali', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_alkaline', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_chalcogen', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_halogen', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_lanthanoid', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_metal', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_metalloid', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_noble_gas', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_post_transition_metal', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_quadrupolar', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_rare_earth_metal', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('experimental_oxidation_states', pa.list_(t_int), metadata={'encoder':'OxidationStatesEncoder(dtype=torch.float32)'}),
    pa.field('ionization_energies', pa.list_(t_float), metadata={'encoder':'IonizationEnergiesEncoder(dtype=torch.float32)'}),
    pa.field('name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('type', t_string, metadata={'encoder':'ClassificationEncoder()'})
    ]

##############################################################################################################################
# chemenv

chemenv_property_schema_list = [
    pa.field('chemenv_name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('coordination', t_int, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('type', t_string, metadata={'encoder':'ClassificationEncoder()'})]


##############################################################################################################################
# Crystal System

crystal_system_property_schema_list = [
    pa.field('crystal_system', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('type', t_string, metadata={'encoder':'ClassificationEncoder()'})]

##############################################################################################################################
# Lattice

lattice_property_schema_list = [
    pa.field('lattice', pa.list_(pa.list_(t_float))),
    pa.field('a', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('b', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('c', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('alpha', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('beta', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('gamma', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('volume', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('type', t_string, metadata={'encoder':'ClassificationEncoder()'})]

##############################################################################################################################
# Sites

site_property_schema_list = [
    pa.field('species', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('frac_coords', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('lattice', pa.list_(pa.list_(t_float))),
    pa.field('material_id', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('type', t_string, metadata={'encoder':'ClassificationEncoder()'})]

##############################################################################################################################
# Magnetic States

magnetic_state_property_schema_list = [
    pa.field('magnetic_state', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('type', t_string, metadata={'encoder':'ClassificationEncoder()'})]

##############################################################################################################################
# Oxidation States

oxidation_state_property_schema_list = [
    pa.field('oxidation_state', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('type', t_string, metadata={'encoder':'ClassificationEncoder()'})]

##############################################################################################################################
# SPG

spg_property_schema_list = [
    pa.field('spg', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('type', t_string, metadata={'encoder':'ClassificationEncoder()'})]


##############################################################################################################################
# spg_wyckoff

spg_wyckoff_property_schema_list = [
    pa.field('spg_wyckoff', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('type', t_string, metadata={'encoder':'ClassificationEncoder()'})]



##############################################################################################################################
# relationship schema


def get_relationship_schema(relationship_type:RelationshipTypes):
    if not isinstance(relationship_type,RelationshipTypes):
        raise ValueError("relationship_type must be an instance of RelationshipTypes.{}")
    node_a_name,connection_type,node_b_name=relationship_type.value.split('-')

    relationship_property_schema_list = [
            pa.field(f'{node_a_name}-START_ID', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
            pa.field(f'{node_b_name}-END_ID', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
            pa.field('TYPE', t_string, metadata={'encoder':'ClassificationEncoder()'}),
            pa.field('weight', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'})]

    return pa.schema(relationship_property_schema_list)

##############################################################################################################################

MATERIAL_PARQUET_SCHEMA = pa.schema(material_property_schema_list)
LATTICE_PARQUET_SCHEMA = pa.schema(lattice_property_schema_list)
SITE_PARQUET_SCHEMA = pa.schema(site_property_schema_list)
ELEMENT_PARQUET_SCHEMA = pa.schema(element_property_schema_list)
CHEMENV_PARQUET_SCHEMA = pa.schema(chemenv_property_schema_list)
MAGNETIC_STATE_PARQUET_SCHEMA = pa.schema(magnetic_state_property_schema_list)
CRYSTAL_SYSTEM_PARQUET_SCHEMA = pa.schema(crystal_system_property_schema_list)
OXIDATION_STATE_PARQUET_SCHEMA = pa.schema(oxidation_state_property_schema_list)
SPG_PARQUET_SCHEMA = pa.schema(spg_property_schema_list)
SPG_WYCKOFF_PARQUET_SCHEMA = pa.schema(spg_wyckoff_property_schema_list)


def get_node_schema(node_type:NodeTypes):
    if not isinstance(node_type,NodeTypes):
        raise ValueError("node_type must be an instance of NodeTypes.{}")
    node_name=node_type.value.lower()
    
    if node_name=='material':
        schema=MATERIAL_PARQUET_SCHEMA
    elif node_name=='element' or node_name=='pre_imputed_element':
        schema=ELEMENT_PARQUET_SCHEMA
    elif node_name=='chemenv':
        schema=CHEMENV_PARQUET_SCHEMA
    elif node_name=='crystal_system':
        schema=CRYSTAL_SYSTEM_PARQUET_SCHEMA
    elif node_name=='magnetic_state':
        schema=MAGNETIC_STATE_PARQUET_SCHEMA
    elif node_name=='space_group':
        schema=SPG_PARQUET_SCHEMA
    elif node_name=='oxidation_state':
        schema=OXIDATION_STATE_PARQUET_SCHEMA
    elif node_name=='spg_wyckoff':
        schema=SPG_WYCKOFF_PARQUET_SCHEMA
    elif node_name=='lattice':
        schema=LATTICE_PARQUET_SCHEMA
    elif node_name=='site':
        schema=SITE_PARQUET_SCHEMA

    return schema
    





# element_property_schema_list = [
#     pa.field('group', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('row', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('Z', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('symbol', t_string, metadata={'encoder':'ClassificationEncoder()'}),
#     pa.field('long_name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
#     pa.field('A', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('atomic_radius_calculated', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('van_der_waals_radius', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('mendeleev_no', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('electrical_resistivity', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('velocity_of_sound', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('reflectivity', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('refractive_index', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('poissons_ratio', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('molar_volume', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('thermal_conductivity', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('boiling_point', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('melting_point', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('critical_temperature', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('superconduction_temperature', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('liquid_range', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('bulk_modulus', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('youngs_modulus', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('rigidity_modulus', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('vickers_hardness', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('density_of_solid', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('coefficient_of_linear_thermal_expansion', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('ionization_energies', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
#     pa.field('block', t_string, metadata={'encoder':'ClassificationEncoder()'}),
#     pa.field('common_oxidation_states', pa.list_(t_int), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
#     pa.field('electron_affinity', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('X', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('atomic_mass', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('atomic_mass_number', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('atomic_radius', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('average_anionic_radius', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('average_cationic_radius', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('average_ionic_radius', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('ground_state_term_symbol', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    
#     pa.field('icsd_oxidation_states', pa.list_(t_int), metadata={'encoder':'IdentityEncoder(dtype=torch.int64)'}),
#     pa.field('is_actinoid', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('is_alkali', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('is_alkaline', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('is_chalcogen', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('is_halogen', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('is_lanthanoid', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('is_metal', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('is_metalloid', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('is_noble_gas', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('is_post_transition_metal', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('is_quadrupolar', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('is_rare_earth', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('is_rare_earth_metal', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('is_transition_metal', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
#     pa.field('iupac_ordering', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('max_oxidation_state', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('min_oxidation_state', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
#     pa.field('oxidation_states', pa.list_(t_int), metadata={'encoder':'ListIdentityEncoder(dtype=torch.int64)'}),
#     pa.field('valence', pa.list_(t_int), metadata={'encoder':'ListIdentityEncoder(dtype=torch.int64)'}),
#     pa.field('name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
#     pa.field('type', t_string, metadata={'encoder':'ClassificationEncoder()'})]

