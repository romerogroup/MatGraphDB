from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
import json
import time

from functools import wraps
from pymongo import MongoClient, ASCENDING

from matgraphdb import DBManager


def load_json_files(directory):
    data = []
    for i,filename in enumerate(os.listdir(directory)):
        if i%1000==0:
            print(i)
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data.append(json.load(file))
    return data


def insert_data(json_data, db_name='MatgraphDB', collection_name='materials'):

    client = MongoClient('mongodb://localhost:27017/') 
    db = client[db_name]

    # Create a collection if it doesn't exist
    collection_names = db.list_collection_names()
    if collection_name not in collection_names:
        db.create_collection(collection_name)
        print(f"Collection {collection_name} created")

    collection = db[collection_name]

    # Insert data into MongoDB
    if isinstance(json_data, list):
        result = collection.insert_many(json_data)
        print(f"Inserted {len(result.inserted_ids)} documents")
    else:
        result = collection.insert_one(json_data)
        print(f"Inserted document with ID: {result.inserted_id}")

    # Close the connection
    client.close()


def update_data(json_data, db_name='MatgraphDB', collection_name='materials'):    
    client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string if needed
    db = client[db_name]
    collection = db[collection_name]

    # Insert data into MongoDB
    if isinstance(json_data, list):
        for doc in json_data:
            query = {"material_id": doc['material_id']}
            update={"$set": doc}
            result = collection.update_one(query, update, upsert=True)
        print(f"Updated documents")
    else:
        query = {"material_id": json_data['material_id']}
        update={"$set": json_data}
        result = collection.update_one(query, update, upsert=True)
        print(f"Updated doc with ID: {json_data['material_id']}")

    # Close the connection
    client.close()

def delete_data(db_name='MatgraphDB', collection_name='materials'):    
    client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string if needed
    db = client[db_name]
    collection = db[collection_name]

    result = collection.delete_many({})

    # Close the connection
    client.close()


def setup_database(db_name='MatgraphDB', collection_name='materials'):
    client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string if needed
    db = client[db_name]
    collection = db[collection_name]

    collection.create_index([('material_id', ASCENDING)], unique=True)

    # Close the connection
    client.close()


# def read_data_task(mongo_cursor):
#     data=[]
#     for post in mongo_cursor:
#         data.append(post)
#     return data

# def read_data(db_name='MatgraphDB', collection_name='materials'):    
#     client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string if needed
#     db = client[db_name]
#     collection = db[collection_name]

#     data=[]
#     for post in collection.find():
#         data.append(post)
#     return data



def read_chunk(start, end, db_name='MatgraphDB', collection_name='materials'):
    client = MongoClient("mongodb://localhost:27017/")
    db = client[db_name]
    collection = db[collection_name]
    
    chunk = list(collection.find().skip(start).limit(end - start))
    client.close()
    return chunk


def read_data(db_name='MatgraphDB', collection_name='materials',num_cores=None):  
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()  
    client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string if needed
    db = client[db_name]
    collection = db[collection_name]

    total_docs = collection.count_documents({})
    chunk_size = total_docs // num_cores

    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_cores)]

    if total_docs % num_cores != 0:
        chunks[-1] = (chunks[-1][0], total_docs)
    client.close()
    print(num_cores)
    # with multiprocessing.Pool(num_cores) as pool:
    with multiprocessing.pool.ThreadPool(num_cores) as pool:
        # results = pool.map(lambda x: read_chunk(*x), chunks)

        results = pool.starmap(read_chunk, chunks)
        # results = list(executor.map(lambda x: read_chunk(*x), chunks))
    data=[]
    for result in results:
        data.extend(result)
    # Flatten the list of chunks
    # all_docs = [doc for chunk in results for doc in chunk]
    return data


def main():

    db_name='MatgraphDB'
    collection_name='materials_db'

    # db_manager=DBManager()
    # start_time=time.time()
    # raw_data=db_manager.load_data()
    # load_time=time.time()-start_time
    # print(f"Load time: {load_time} seconds")
    # data=[]
    # for mat_data in raw_data:
    #     if isinstance(mat_data,dict):
    #         data.append(mat_data)

    
    # delete_data(db_name=db_name, collection_name=collection_name)

    # setup_database(db_name=db_name, collection_name=collection_name)
    # insert_data(data, db_name=db_name,collection_name=collection_name)
    # data=None

    # # update_data(data, db_name=db_name,collection_name=collection_name)

    # # delete_data(db_name='MatgraphDB', collection_name='materials')

    start_time=time.time()
    read_data(db_name=db_name, collection_name=collection_name)
    load_time=time.time()-start_time
    print(f"Load time: {load_time} seconds")



    # client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string if needed

    # print(dir(client))
    # db = client[db_name]  # Replace with your database name
    # print(dir(db))
    # collection = db[collection_name]  # Replace with your collection name
    # print(client.list_database_names())
    # print(dir(collection))
    # # Close the connection



    # for post in collection.find():
    #     print(post)
    # client.close()
    

    


if __name__ == '__main__':
    main()




