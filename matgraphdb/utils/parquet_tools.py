import os

from parquetdb import ParquetDB


def write_schema_summary(materials_parquetdb_path, output_path):

    db = ParquetDB(db_path=materials_parquetdb_path)
    table = db.read()
    print(table.shape)

    dir_path = os.path.dirname(output_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(output_path), "w") as f:
        f.write(f"Number of rows: {table.shape[0]}\n")
        f.write(f"Number of columns: {table.shape[1]}\n\n")
        f.write("-" * 100 + "\n\n")

        f.write(f"{'Field Name':<50} | {'Field Type'}\n")
        f.write("-" * 50 + "\n")
        for field in table.schema:
            f.write(f"{field.name:<50} | {field.type}\n")
