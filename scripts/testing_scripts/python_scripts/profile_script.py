import os

import cProfile
import pstats


def main():
    from matgraphdb.graph_kit.node_types import NodeTypes
    nodes=NodeTypes()
    names=nodes.get_chemenv_element_nodes()
    # print(names[:10])


if __name__ == "__main__":
    from matgraphdb.utils import ROOT
    profiler = cProfile.Profile()
    profiler.enable()
    print(ROOT)
    # Run the function you want to profile
    main()
    
    profiler.disable()
    
    # Save the stats to a file
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime')
    stats.dump_stats(os.path.join(ROOT,'logs','profile_results.prof'))  # Save to .prof file
    with open(os.path.join(ROOT,'logs','profile_results.txt'), 'w') as f:
        stats = pstats.Stats(os.path.join(ROOT,'logs','profile_results.prof'), stream=f)
        stats.sort_stats('tottime')
        stats.print_stats()
