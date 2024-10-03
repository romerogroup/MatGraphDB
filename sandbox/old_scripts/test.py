

def print_memory_usage(message):
    import psutil
    # Get the memory usage of the current process in MB
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"{message}: {memory_info.rss / (1024 * 1024):.2f} MB")


def use_pyarrow():
    # from matgraphdb import DBManager
    import pyarrow as pa
    import pyarrow.parquet as pq
    import time
    

    print_memory_usage("Before creating PyArrow Table")

    table = pq.read_table('/users/lllang/SCRATCH/projects/MatGraphDB/data/production/materials_project/materials_database.parquet')
    

    print_memory_usage("After creating PyArrow Table")

    column_names = table.column_names
    print(column_names)

    df = table.to_pandas()

    print_memory_usage("After converting to Pandas DataFrame")

    # Find duplicates in the 'material_id' column
    duplicates = df[df.duplicated('material_id', keep=False)]

    print_memory_usage("After finding duplicates")

    # Print the duplicates with their indices
    duplicates_with_index = duplicates[['material_id']].reset_index()

    print("Duplicates in 'material_id' column with indices:")
    print(duplicates_with_index)


    print_memory_usage("After printing duplicates")



    # start_time=time.time()
    # material_id_column=table['sine_coulomb_matrix']
    # print(material_id_column)
    # print("End Time: ", time.time() - start_time)


    # start_time=time.time()
    # feature_vectors=table['feature_vectors']
    # sine_coulomb_matrix=[]
    # for feature_vector in feature_vectors:
    #     tmp_dict=feature_vector.get('sine_coulomb_matrix')
    #     if tmp_dict is not None:
    #         sine_coulomb_matrix.append(tmp_dict.get('values'))
    # print("End Time: ", time.time() - start_time)
    
def main():
    pass


if __name__ == '__main__':
    use_pyarrow()

    # main()