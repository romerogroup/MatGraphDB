# import os
# from poly_graphs_lib.utils import PROJECT_DIR

# def main():
#     # nodes_dir=os.path.join(PROJECT_DIR,'poly_graphs_lib','database','populate','nodes')

#     # script_names=[
#     #     'chemenv_nodes.py',
#     #     'crystal_system_nodes.py',
#     #     'element_nodes.py',
#     #     'magnetic_state_nodes.py',
#     #     'material_nodes.py',
#     #     'space_group_number_nodes.py',
#     # ]

#     # script_statements=[os.path.join(nodes_dir,name) for name in script_names]

#     # # script_statements=[f"python {name}" for name in script_names]
    
#     # # for script in script_statements:
#     # #     exec(script)

#     # for script in script_statements:
#     #     result = runpy.run_path(script)

#     # This statement Connects to the database server
#     # connection = GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD))
#     # # To read and write to the data base you must open a session
#     # session = connection.session(database=DB_NAME)

# if __name__ =='__main__':
#     main()