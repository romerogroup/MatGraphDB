# TODO


## Phase 1
- [ ] Make MatGraphDB more user freindly with documentation
    - **difficulty** - Easy 
    - [x] Refactor repository Structure
    - [ ] Add better description to the README.md on how to use package
    - [ ] Add documentation to package
    - [ ] Add docstrings to all methods.
- [ ] Python functions to screen graph 
    - **difficulty** - Easy 
    - This would involvde python function that use Cypher queries on the graph
- [ ] Material Similarity Network
    - [x] Produce algorithm that will calculate the similarity based on Encoding then store the results
    - **difficulty** - Medium-somewhat Hard
    - ~79,000 nodes, ~3,120,500,000 similarity scores
    - single-precision (4 bytes per float) -> 12.482 GB
    - double-precision (8 bytes per float) -> 24.964 GB
    - Assuming json storage in the format {mp-1000_mp-1001:0.900}
        - Assuming ACII (1 byte per character) : key-values (Assuming 15 char-avg) -> 15\*1\*3,120,500,000 = 46.808 GB
        - Assuming ACII (1 byte per character) : Take account of `'` and `:` -> 2\*1\*3,120,500,000 = 6.241 GB
    - To use these values in an algorithm I would need a minimium of `46.808 GB + 6.241 GB + 12.482 GB = 65.877 GB` RAM 
    - To use in graph algorith such as the package networkx, I would need to load the json into memory then load relationships into  (>=12 GB)
    - I need to find a way to load the data in by relationship this would reduce the need to load it all in at once.
- [ ] Generate Encodings from different material fingerprint methods
    - **difficulty** - Medium-somewhat Hard
    - [ ] Crystal Diffusion Variational AutoEncoder
        - **difficulty** - Medium-somewhat Hard
    - [ ] Crabnet
        - **difficulty** - Somewhat Easy
    - [ ] MEGnet
        - **difficulty** - Easy
    - [ ] Matminer
        - **difficulty** - Easy
    - [ ] Text Embedding of material OpenAI embedding 
    - **difficulty** - Somewhat Easy
- [ ] Start Writing Paper
    - **difficulty** - Easy if I accomplish all previous task

## Phase 2
- [ ] Graphical User Interface GUI
    - **difficulty** Medium-somewhat Hard
- [ ] Text-To-Text LLM Integration
    - **difficulty** Somewhat Easy - Somewhat Hard
- [ ] Predict missing material properties
    - **difficulty** - Medium-somewhat Hard
- [ ] Can we predict different alloys
    - **difficulty** - Hard
- [ ] Can we predict new materials from the ChemEnv polyhedral environments
    - **difficulty**  Hard
    - Inferred relationships?
- [ ] Store Density of States from -5 to 5
    - **difficulty** - Easy

## ONGOING 
- [x] More Efficient way to perform DFT Calculations
    - previously I seperated the database into binary, ternary, and quatnary in to separate directories. This is a dumb way to do. I should have a single directory with all materials and just aggregate after.
- [ ] Finish chargemol calculations
    - Still running issues on cluster