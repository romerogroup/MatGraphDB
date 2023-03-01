import os

from poly_graphs_lib.data.create_datasets import create_plutonic_dataset


def main():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.dirname(project_dir) + os.sep + 'data'

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)


    feature_set_indices = [0,1,2,3,4,5]
    val_size=0.20

    for feature_set_index in feature_set_indices:
       create_plutonic_dataset(data_dir=dataset_dir, 
                                feature_set_index=feature_set_index, 
                                val_size=val_size)

if __name__ == '__main__':
    main()