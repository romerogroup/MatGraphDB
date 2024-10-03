from matgraphdb import DBManager

def main():
    manager=DBManager()
    manager.generate_wyckoff_positions()
    
if __name__ == '__main__':
    main()