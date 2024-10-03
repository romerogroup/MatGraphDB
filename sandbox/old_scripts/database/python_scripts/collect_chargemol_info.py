from matgraphdb import DBManager

def main():
    manager=DBManager()
    manager.collect_chargemol_info()
    
if __name__ == '__main__':
    main()