import pandas as pd



# from matminer.featurizers.composition import ElementProperty, ElementFraction
# if __name__ == '__main__':
        
#     feature_dict = {
#         'formula':[]
#         'property':[]
#     }

#     for material_file in materials_files:
#         feature_dict['formula'].append(composition.formula)   
#         feature_dict['property'].append(0.0) # This is a placeholder for the property value


#     df=pd.DataFrame(feature_dict)

#     df.head()

#     from matminer.featurizers.conversions import StrToComposition
#     df = StrToComposition().featurize_dataframe(df, "formula")
#     df.head()

#     MultipleFeaturizer([ElementFraction(),ElementProperty.from_preset(preset_name="magpie")])
#     ep_feat = ElementProperty.from_preset(preset_name="magpie")
#     df = ep_feat.featurize_dataframe(df, col_id="composition")  # input the "composition" column to the featurizer
#     df.head()
#     composition_features.values[0,[1,3:]]


from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# Silicon diamond structure constants
a = 5.431  # Lattice constant for Silicon
diamond_lattice = Lattice.cubic(a)

# Positions of the atoms in the unit cell
positions = [(0, 0, 0), (0.25, 0.25, 0.25)]  # Basis atoms for diamond structure

# Create the structure
silicon_diamond_structure = Structure(diamond_lattice, ["Si", "Si"], positions)



# Write the structure to a CIF format string
cif_writer = CifWriter(silicon_diamond_structure)
cif_text = cif_writer.__str__()

print(cif_text)
