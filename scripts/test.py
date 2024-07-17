from matgraphdb import DBManager
import pyarrow as pa
import pyarrow.parquet as pq
import time
def main():
    
    table = pq.read_table('/users/lllang/SCRATCH/projects/MatGraphDB/data/production/materials_project/materials_database.parquet')
    
    start_time=time.time()
    material_id_column=table['sine_coulomb_matrix']
    print(material_id_column)
    print("End Time: ", time.time() - start_time)


    # start_time=time.time()
    # feature_vectors=table['feature_vectors']
    # sine_coulomb_matrix=[]
    # for feature_vector in feature_vectors:
    #     tmp_dict=feature_vector.get('sine_coulomb_matrix')
    #     if tmp_dict is not None:
    #         sine_coulomb_matrix.append(tmp_dict.get('values'))
    # print("End Time: ", time.time() - start_time)
    
    
if __name__ == '__main__':
    main()