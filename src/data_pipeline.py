from poly_graphs_lib.data.data_collection import PolyCollector
from poly_graphs_lib.data.data_featurization import FeatureGenerator
from poly_graphs_lib.data.pair_generation import PairGenerator
from poly_graphs_lib.data.regression_generation import RegressionGenerator

def main():
    collect_raw_polys = False
    generate_features = True
    generate_pairs = False
    generate_regression= False
    
    # Generate poly data
    if collect_raw_polys:
        data_generator = PolyCollector()
        data_generator.initialize_ingestion()

    
    if generate_features:
        data_generator = FeatureGenerator()
        data_generator.initialize_generation()


    # For Regression-like model 
    if generate_regression:
        data_generator = RegressionGenerator()
        data_generator.initialize_generation()

    # For Siamese-like model 
    if generate_pairs:
        data_generator = PairGenerator()
        data_generator.initialize_generation()

if __name__ == '__main__':
    main()