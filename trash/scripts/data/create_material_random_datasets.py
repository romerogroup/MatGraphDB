import os

from matgraphdb.data.create_datasets import create_material_random_polyhedra_dataset


def main():
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.dirname(parent_dir) + os.sep + 'datasets'
    print(dataset_dir)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    mpcif_data_dir = f'/users/lllang/SCRATCH/crystal_gen/Graph_Network_Project/datasets/raw/mp_cif'

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