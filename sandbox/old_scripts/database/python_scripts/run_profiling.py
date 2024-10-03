import cProfile
import pstats


def main():
    from matgraphdb.graph_kit.graph_generator import GraphGenerator

    generator=GraphGenerator(from_scratch=False)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the function you want to profile
    main()
    
    profiler.disable()
    
    # Save the stats to a file
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime')
    stats.dump_stats('logs/profile_results.prof')  # Save to .prof file
    with open('logs/profile_results.txt', 'w') as f:
        stats = pstats.Stats('logs/profile_results.prof', stream=f)
        stats.sort_stats('tottime')
        stats.print_stats()
