import pandas as pd



from matminer.featurizers.composition import ElementProperty, ElementFraction



if __name__ == '__main__':
        
    feature_dict = {
        'formula':[]
        'property':[]
    }

    for material_file in materials_files:
        feature_dict['formula'].append(composition.formula)   
        feature_dict['property'].append(0.0) # This is a placeholder for the property value


    df=pd.DataFrame(feature_dict)

    df.head()

    from matminer.featurizers.conversions import StrToComposition
    df = StrToComposition().featurize_dataframe(df, "formula")
    df.head()

    MultipleFeaturizer([ElementFraction(),ElementProperty.from_preset(preset_name="magpie")])
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition")  # input the "composition" column to the featurizer
    df.head()
    composition_features.values[0,[1,3:]]