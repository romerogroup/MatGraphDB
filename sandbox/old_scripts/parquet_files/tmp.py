import os
import pyarrow as pa
import pyarrow.parquet as pq



# dataset_dir='data/raw/ParquetDB/datasets/main'
# # Read the Parquet file into a PyArrow table
# table_path=os.path.join(dataset_dir,'main_0.parquet')
# table = pq.read_table(table_path)

# print(table.column_names)
# # # Get the column by name (assume we are modifying 'column_name')
# column_to_modify = table['band_gap']

# # # Convert the column to a list to modify values (for example, let's change the first value)
# column_list = column_to_modify.to_pylist()
# column_list[0] = 'new_value'

# # Convert the modified list back to a PyArrow array
# new_column = pa.array(column_list)

# i=0
# for i,column in enumerate(table.itercolumns()):
#     print(column._name)
#     # if column._name=='id':
#     #     index=column.index([0,1])
#     #     print(index)
#     if column._name=='band_gap':
#         print(column)
#         print(dir(column))
#         print(column.to_pylist())
#         flatten_array=column.to_pylist()
#         # flatten_array=column.combine_chunks().to_pylist()
#         flatten_array[0:5]=[5]*5
# #         print(len(flatten_array))
# #         # print(dir(column[0]))
# #         # print(flatten_array)

#         table=table.set_column(i,column._name, [flatten_array])

# print(table['band_gap'])
# # Replace the column using set_column (index, new_name, new_column)
# new_table = table.set_column(table.schema.get_field_index('column_name'), 'column_name', new_column)

# # Optionally, write the modified table back to a new Parquet file
# pq.write_table(new_table, 'modified_example.parquet')

# # Display the new table
# print(new_table)


# from multiprocessing import Pool

# def worker_2(item):
#     # Simulate some processing
#     return item * 2

# def worker_1(data):
#     with Pool(3) as p:
#         results = p.map(worker_2, data)
#     return results

# def main(data_list_1):
#     with Pool(2) as p:
#         results = p.map(worker_1, data_list_1)
#     return results

# if __name__ == "__main__":
#     # Example usage
#     data_list_1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
#     final_results = main(data_list_1)
#     print(final_results)

incoming_data = set(['c',])
current_data = set(['a', 'b'])

print(incoming_data-current_data)