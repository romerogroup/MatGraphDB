import os

from poly_graphs_lib.data.create_datasets import create_material_random_polyhedra_dataset


def main():
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.dirname(parent_dir) + os.sep + 'data'

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    mpcif_data_dir = f'C:/Users/lllang/Desktop/Romero Group Research/Research Projects/crystal_generation_project/Voronoi_Project/datasets/mp_cif'

    feature_set_indices = [0,1,2,3,4,5]
    val_size=0.15

    for feature_set_index in feature_set_indices:
       create_material_random_polyhedra_dataset(data_dir=dataset_dir,
                                                mpcif_data_dir=mpcif_data_dir,
                                                feature_set_index=feature_set_index, 
                                                n_polyhedra=1000,
                                                val_size=val_size)

if __name__ == '__main__':
    main()