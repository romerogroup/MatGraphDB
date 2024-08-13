import pyarrow as pa
from matgraphdb.graph.types import NodeTypes, RelationshipTypes

t_string=pa.string()
t_int=pa.int64()
t_float=pa.float64()
t_bool=pa.bool_()

material_property_schema_list=[
    pa.field('material_id', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('nsites', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elements', pa.list_(t_string), metadata={'encoder':'ElementsEncoder(dtype=torch.float32)'}),
    pa.field('nelements', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('formula_pretty', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('volume', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('density', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('density_atomic', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('crystal_system', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('space_group', t_int, metadata={'encoder':'SpaceGroupOneHotEncoder()'}),
    pa.field('space_group_symbol', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('point_group', t_string, metadata={'encoder':'ClassificationEncoder()'}),

    pa.field('composition-values', pa.list_(t_float)),
    pa.field('composition-elements', pa.list_(t_string)),
    pa.field('composition_reduced-values', pa.list_(t_float)),
    pa.field('composition_reduced-elements', pa.list_(t_string)),

    pa.field('lattice',pa.list_(pa.list_(t_float))),
    # pa.field('pbc', pa.list_(t_bool)),
    pa.field('a', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('b', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('c', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('alpha', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('beta', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('gamma', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('frac_coords', pa.list_(pa.list_(t_float))),
    pa.field('cart_coords', pa.list_(pa.list_(t_float))),
    pa.field('species', pa.list_(t_string)),
    pa.field('unit_cell_volume', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),

    pa.field('energy_per_atom', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('formation_energy_per_atom', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('energy_above_hull', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('band_gap', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('cbm', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('vbm', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('efermi', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('is_stable', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_gap_direct', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_metal', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_magnetic', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('ordering', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('total_magnetization', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('total_magnetization_normalized_vol', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('num_magnetic_sites', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('num_unique_magnetic_sites', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('k_voigt', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('k_reuss', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('k_vrh', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('g_voigt', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('g_reuss', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('g_vrh', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('universal_anisotropy', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('homogeneous_poisson', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('e_total', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('e_ionic', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('e_electronic', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),

    pa.field('sine_coulomb_matrix', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('element_fraction', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('element_property', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('xrd_pattern', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),

    pa.field('coordination_environments_multi_weight', pa.list_(pa.list_(pa.struct([('ce_symbol', t_string),
                                                                    ('ce_fraction', t_float),
                                                                    ('csm', t_float),
                                                                    ('permutation', pa.list_(t_int))])))),
    pa.field('coordination_multi_connections', pa.list_(pa.list_(t_int))),
    pa.field('coordination_multi_numbers', pa.list_(t_int), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('chargemol_bonding_connections', pa.list_(pa.list_(t_int))),
    pa.field('chargemol_bonding_orders', pa.list_(pa.list_(t_float)), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    # pa.field('chargemol_net_atomic_charges', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('chargemol_squared_moments', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('chargemol_cubed_moments', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('chargemol_fourth_moments', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('geometric_consistent_bond_connections', pa.list_(pa.list_(t_int))),
    pa.field('geometric_consistent_bond_orders', pa.list_(pa.list_(t_float)), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('electric_consistent_bond_connections', pa.list_(pa.list_(t_int))),
    pa.field('electric_consistent_bond_orders', pa.list_(pa.list_(t_float)), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('geometric_electric_consistent_bond_connections', pa.list_(pa.list_(t_int))),
    pa.field('geometric_electric_consistent_bond_orders', pa.list_(pa.list_(t_float)), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('bond_cutoff_connections', pa.list_(pa.list_(t_int))),
    pa.field('wyckoffs', pa.list_(t_string)),

    pa.field('oxidation_states-possible_species', pa.list_(t_string)),
    pa.field('oxidation_states-possible_valences', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('oxidation_states-method', t_string, metadata={'encoder':'ClassificationEncoder()'}),

    pa.field('last_updated', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('uncorrected_energy_per_atom', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('equilibrium_reaction_energy_per_atom', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('decomposes_to', pa.list_(pa.struct([
                                ('amount', t_float),
                                ('formula', t_string),
                                ('material_id', t_string)]))),
    pa.field('types_of_magnetic_species', pa.list_(t_string)),
    pa.field('grain_boundaries', pa.list_(pa.struct([
                                ('gb_energy', t_float),
                                ('rotation_angle', t_float),
                                ('sigma', t_int),
                                ('type', t_string)]))),
    pa.field('dos_energy_up', pa.list_(t_float), metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),

    pa.field('n', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('e_ij_max', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('weighted_surface_energy_EV_PER_ANG2', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('weighted_surface_energy', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('weighted_work_function', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('surface_anisotropy', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('shape_factor', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('has_reconstructed', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('possible_species', pa.list_(t_string)),
    pa.field('theoretical', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-materials', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-thermo', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-xas', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-grain_boundaries', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-chemenv', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-electronic_structure', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-absorption', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-bandstructure', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-dos', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-magnetism', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-elasticity', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-dielectric', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-piezoelectric', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-surface_properties', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-oxi_states', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-provenance', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-charge_density', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-eos', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-phonon', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-insertion_electrodes', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-substrates', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('elasticity-warnings', pa.list_(t_string)),
    pa.field('elasticity-order', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-k_vrh', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-k_reuss', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-k_voigt', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-g_vrh', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-g_reuss', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-g_voigt', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-sound_velocity_transverse', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-sound_velocity_longitudinal', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-sound_velocity_total', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-sound_velocity_acoustic', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-sound_velocity_optical', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-thermal_conductivity_clarke', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-thermal_conductivity_cahill', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-young_modulus', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-universal_anisotropy', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-homogeneous_poisson', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-debye_temperature', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-state', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('type', t_string, metadata={'encoder':'ClassificationEncoder()'})]

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