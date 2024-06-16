from matgraphdb.graph.graph_generator import GraphGenerator

def main():
    generator=GraphGenerator(from_scratch=False)

    generator.screen_graph_database('nelements-2-2',nelements=(2,2), from_scratch=True)
    generator.screen_graph_database('nelements-3-3',nelements=(3,3), from_scratch=True)
    generator.screen_graph_database('nelements-4-4',nelements=(4,4), from_scratch=True)
    generator.screen_graph_database('nelements-5-5',nelements=(5,5), from_scratch=True)

    generator.screen_graph_database('nelements-2-3',nelements=(1,3), from_scratch=True)
    generator.screen_graph_database('nelements-2-4',nelements=(1,4), from_scratch=True)
    generator.screen_graph_database('nelements-2-5',nelements=(1,5), from_scratch=True)

    generator.screen_graph_database('nelements-3-4',nelements=(3,4), from_scratch=True)
    generator.screen_graph_database('nelements-3-5',nelements=(3,5), from_scratch=True)
    
    generator.screen_graph_database('nelements-4-5',nelements=(4,5), from_scratch=True)

    generator.screen_graph_database('spg-145',space_groups=[145], from_scratch=True)
    generator.screen_graph_database('spg-145-196',space_groups=[145,196], from_scratch=True)
    generator.screen_graph_database('spg-no-145',space_groups=[145], from_scratch=True, include=False)
    generator.screen_graph_database('spg-no-196',space_groups=[196], from_scratch=True, include=False)

    generator.screen_graph_database('elements-no-Ti',elements=["Ti"], from_scratch=True, include=False)
    generator.screen_graph_database('elements-no-Fe',elements=["Fe"], from_scratch=True, include=False)
    generator.screen_graph_database('elements-no-Ti-Fe',elements=["Ti","Fe"], from_scratch=True, include=False)

if __name__ == '__main__':
    main()