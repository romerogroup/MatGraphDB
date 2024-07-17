from matgraphdb import DBManager

def main():
    manager=DBManager()
    manager.create_parquet_file()
    
    
if __name__ == '__main__':
    main()