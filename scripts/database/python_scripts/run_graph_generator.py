from matgraphdb.graph.graph_generator import GraphGenerator

def main():
    generator=GraphGenerator(from_scratch=False)
    generator.screen_graph_database('nelements_1-2',nelements=(1,2), from_scratch=True)
    
if __name__ == '__main__':
    main()