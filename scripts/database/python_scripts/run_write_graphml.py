import os
from matgraphdb.graph_kit.graph_generator import GraphGenerator

def main():
    generator=GraphGenerator()
    main_graph_dir = generator.main_graph_dir
    graph_dir= os.path.join(main_graph_dir,'sub_graphs','nelements-2-2')
    generator.write_graphml(graph_dirname=graph_dir)

if __name__ == '__main__':
    main()