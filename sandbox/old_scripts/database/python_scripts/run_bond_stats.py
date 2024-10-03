from matgraphdb import DBManager

def main():
    manager=DBManager()
    manager.bond_orders_stats_calculation()
    
if __name__ == '__main__':
    main()
