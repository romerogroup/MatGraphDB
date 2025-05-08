matgraphdb.materials.nodes.materials.MaterialStore
==================================================

.. currentmodule:: matgraphdb.materials.nodes.materials

.. autoclass:: MaterialStore

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~MaterialStore.__init__
      ~MaterialStore.backup_database
      ~MaterialStore.construct_table
      ~MaterialStore.copy_dataset
      ~MaterialStore.create
      ~MaterialStore.create_material
      ~MaterialStore.create_materials
      ~MaterialStore.create_nodes
      ~MaterialStore.dataset_exists
      ~MaterialStore.delete
      ~MaterialStore.delete_materials
      ~MaterialStore.delete_nodes
      ~MaterialStore.drop_dataset
      ~MaterialStore.export_dataset
      ~MaterialStore.export_partitioned_dataset
      ~MaterialStore.get_current_files
      ~MaterialStore.get_field_metadata
      ~MaterialStore.get_field_names
      ~MaterialStore.get_file_sizes
      ~MaterialStore.get_metadata
      ~MaterialStore.get_n_rows_per_row_group_per_file
      ~MaterialStore.get_number_of_row_groups_per_file
      ~MaterialStore.get_number_of_rows_per_file
      ~MaterialStore.get_parquet_column_metadata_per_file
      ~MaterialStore.get_parquet_file_metadata_per_file
      ~MaterialStore.get_parquet_file_row_group_metadata_per_file
      ~MaterialStore.get_row_group_sizes_per_file
      ~MaterialStore.get_schema
      ~MaterialStore.get_serialized_metadata_size_per_file
      ~MaterialStore.import_dataset
      ~MaterialStore.initialize
      ~MaterialStore.is_empty
      ~MaterialStore.merge_datasets
      ~MaterialStore.normalize
      ~MaterialStore.normalize_nodes
      ~MaterialStore.preprocess_table
      ~MaterialStore.process_data_with_python_objects
      ~MaterialStore.read
      ~MaterialStore.read_materials
      ~MaterialStore.read_nodes
      ~MaterialStore.rename_dataset
      ~MaterialStore.rename_fields
      ~MaterialStore.restore_database
      ~MaterialStore.set_field_metadata
      ~MaterialStore.set_metadata
      ~MaterialStore.sort_fields
      ~MaterialStore.summary
      ~MaterialStore.to_nested
      ~MaterialStore.transform
      ~MaterialStore.update
      ~MaterialStore.update_materials
      ~MaterialStore.update_nodes
      ~MaterialStore.update_schema
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~MaterialStore.basename_template
      ~MaterialStore.columns
      ~MaterialStore.dataset_name
      ~MaterialStore.db_path
      ~MaterialStore.n_columns
      ~MaterialStore.n_features
      ~MaterialStore.n_files
      ~MaterialStore.n_nodes
      ~MaterialStore.n_row_groups_per_file
      ~MaterialStore.n_rows
      ~MaterialStore.n_rows_per_file
      ~MaterialStore.n_rows_per_row_group_per_file
      ~MaterialStore.name_column
      ~MaterialStore.node_metadata_keys
      ~MaterialStore.serialized_metadata_size_per_file
      ~MaterialStore.storage_path
   
   