from matgraphdb import DBManager

def main():
    manager=DBManager()
    print(len(manager.database_files()))
    manager.chemenv_calc()
    
if __name__ == '__main__':
    main()