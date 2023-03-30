from data.data_collection import PolyCollector

def main():


    collect_raw_polys = True
    # Generate poly data
    if collect_raw_polys:
        data_generator = PolyCollector()
        data_generator.initialize_ingestion()

if __name__ == '__main__':
    main()