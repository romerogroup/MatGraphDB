import math

from matgraphdb.graph_kit.neo4j.neo4j_manager import Neo4jManager
from matgraphdb.graph_kit.neo4j.neo4j_gds_manager import Neo4jGDSManager
from matgraphdb.graph_kit.neo4j.utils import format_dictionary

def shannon_entropy(probabilities):
    """
    Calculates the Shannon entropy of a list of probabilities.

    Args:
        probabilities (list): A list of probabilities.

    Returns:
        float: The Shannon entropy.
    """
    probabilities = [p for p in probabilities if p > 0]
    total_probability = sum(probabilities)
    entropy=0
    for p in probabilities:
        entropy += - p * math.log(p, 2)
    return entropy

class Neo4jAnalyzer:
    def __init__(self,neo4j_manager=Neo4jManager()):

        self.neo4j_manager=neo4j_manager
    
    def get_node_degrees(self,database_name):
        with self.neo4j_manager as session:
            results=session.query(f"MATCH (n) RETURN n.name AS name",database_name)

            name=results[0]['name']
            prop_dict={'name':name}
            degrees_dict={}
            node_count=len(results)
            for result in results:
                name=result['name']
                prop_dict={'name':name}
                cypher_statement=f"MATCH (n {format_dictionary(prop_dict)})-[r]-() RETURN n.name, COUNT(r) AS degree"
                degree_results=session.query(cypher_statement,database_name)
                if len(degree_results)==0:
                    degree=0
                else:
                    degree=degree_results[0]['degree']
                
                tmp_dict={'degree':degree,'degree_probability':degree/node_count}
                degrees_dict[name]=tmp_dict
            return degrees_dict
        
if __name__=='__main__':
    analyzer=Neo4jAnalyzer()
    degrees_dict=analyzer.get_node_degrees('nelements-1-2')
    degree_probabilities=[degrees_dict[name]['degree_probability'] for name in degrees_dict]
    print("Shannon Entropy: ",shannon_entropy(degree_probabilities))