from matgraphdb.graph.graph_generator import GraphGenerator

def main():
    graph=GraphGenerator()
    graph.write_graphml(graph_dirname='nelements-2-2')

if __name__ == '__main__':
    main()