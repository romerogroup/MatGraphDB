import json

def filter_bond_indices(bond_orders, threshold=0.1):
    """
    Filter bond indices based on a threshold value.
    
    Args:
        bond_orders (list): List of bond orders.
        threshold (float): Threshold value for filtering.
    
    Returns:
        list: List of indices for bond orders above the threshold.
    """
    return [i for i, order in enumerate(bond_orders) if order > threshold]

def adjust_bond_connections_and_orders(geo_coord_connections, elec_coord_connections, bond_orders):
    """
    Adjusts bond connections and orders to be consistent between geometric and electric representations.

    Args:
        geo_coord_connections (list): Geometric bond connections.
        elec_coord_connections (list): Electric bond connections.
        bond_orders (list): List of bond orders.

    Returns:
        tuple: Adjusted electric bond connections and bond orders.
    """
    adjusted_connections = []
    adjusted_bond_orders = []

    for geo_connections, elec_connections, orders in zip(geo_coord_connections, elec_coord_connections, bond_orders):
        indices = filter_bond_indices(orders)
        adjusted_connections.append([elec_connections[i] for i in indices])
        adjusted_bond_orders.append([orders[i] for i in indices])

    return adjusted_connections, adjusted_bond_orders

def process_file(file_path, from_scratch=False):
    """
    Process a file to adjust bond connections and orders.

    Args:
        file_path (str): Path to the file to process.
        from_scratch (bool): Process from scratch regardless of existing data.
    """
    try:
        with open(file_path, 'r') as file:
            db = json.load(file)

        if 'adjusted_bond_connections' not in db or from_scratch:
            geo_connections = db.get('coordination_multi_connections', [])
            elec_connections = db.get('chargemol_bonding_connections', [])
            bond_orders = db.get('chargemol_bonding_orders', [])

            adjusted_connections, adjusted_orders = adjust_bond_connections_and_orders(geo_connections, elec_connections, bond_orders)

            db['adjusted_bond_connections'] = adjusted_connections
            db['adjusted_bond_orders'] = adjusted_orders

            with open(file_path, 'w') as file:
                json.dump(db, file, indent=4)
                
    except Exception as e:
        print(f"Error processing file {file_path.split('/')[-1]}: {e}")

def main_process():
    """
    Main process to run the bond adjustment calculation.
    """
    file_paths = ["path/to/your/file.json"]  # Example file paths
    for file_path in file_paths:
        process_file(file_path, from_scratch=True)

# Example usage

if __name__ == "__main__":

    main_process()
