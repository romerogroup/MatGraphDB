

from matgraphdb.utils import LOGGER
from matgraphdb.graph_kit.metadata import NodeTypes
from matgraphdb.graph_kit.neo4j.neo4j_manager import Neo4jManager
from matgraphdb.graph_kit.neo4j.neo4j_gds_manager import Neo4jGDSManager


class GraphProjection:
    def __init__(self, 
                 name, 
                 node_projections, 
                 relationship_projections, 
                 write_property, 
                 use_weights, 
                 use_node_properties):
        self.name = name
        self.node_projections = node_projections
        self.relationship_projections = relationship_projections
        self.write_property = write_property
        self.use_weights = use_weights
        self.use_node_properties = use_node_properties
        node_properties=[]
        if use_node_properties:
            for key,value in self.node_projections.items():
                if 'properties' in value:
                    node_properties.append(value['properties'])
        self.node_properties=node_properties

    @classmethod
    def ec_element_chemenv(cls, use_weights=True, use_node_properties=True,algorithm_name='fastRP'):
        relationship_projections={
                "`CHEMENV-ELECTRIC_CONNECTS-CHEMENV`": {"orientation": 'UNDIRECTED', "properties": 'weight'},
                "`ELEMENT-ELECTRIC_CONNECTS-ELEMENT`": {"orientation": 'UNDIRECTED', "properties": 'weight'},
                "`CHEMENV-CAN_OCCUR-ELEMENT`": {"orientation": 'UNDIRECTED', "properties": 'weight'},
                "`MATERIAL-HAS-ELEMENT`": {"orientation": 'UNDIRECTED', "properties": 'weight'},
                "`MATERIAL-HAS-CHEMENV`": {"orientation": 'UNDIRECTED', "properties": 'weight'}
            }
        node_projections={
                "Chemenv": {"label":'Chemenv'},
                "Element": {"label":'Element', 
                            "properties":{
                                'atomic_number':{'default_value':0.0},
                                'X':{'default_value':0.0},
                                'atomic_radius':{'default_value':0.0},
                                'group':{'default_value':0},
                                'row':{'default_value':0},
                                'atomic_mass':{'default_value':0.0}
                                }
                            },
                "Material":{"label":'Material'}
            }
        
        write_property=f'{algorithm_name}-embedding-ec-element-chemenv'

        if use_weights:
            write_property+='-weighted'
        if use_node_properties:
            write_property+='-node_properties'
        return cls(
            name="EC Element ChemEnv",
            node_projections=node_projections,
            relationship_projections=relationship_projections,
            write_property=write_property,
            use_weights=use_weights,
            use_node_properties=use_node_properties
        )

    @classmethod
    def gc_element_chemenv(cls, use_weights=True, use_node_properties=True,algorithm_name='fastRP'):
        relationship_projections={
            "`CHEMENV-GEOMETRIC_CONNECTS-CHEMENV`": {"orientation": 'UNDIRECTED', "properties": 'weight'},
            "`ELEMENT-GEOMETRIC_CONNECTS-ELEMENT`": {"orientation": 'UNDIRECTED', "properties": 'weight'},
            "`CHEMENV-CAN_OCCUR-ELEMENT`": {"orientation": 'UNDIRECTED', "properties": 'weight'},
            "`MATERIAL-HAS-ELEMENT`": {"orientation": 'UNDIRECTED', "properties": 'weight'},
            "`MATERIAL-HAS-CHEMENV`": {"orientation": 'UNDIRECTED', "properties": 'weight'},
        }
        node_projections={
                "Chemenv": {"label":'Chemenv'},
                "Element": {"label":'Element',
                            "properties":{
                                'atomic_number':{'default_value':0.0},
                                'X':{'default_value':0.0},
                                'atomic_radius':{'default_value':0.0},
                                'group':{'default_value':0},
                                'row':{'default_value':0},
                                'atomic_mass':{'default_value':0.0}
                                } 
                            },
                "Material":{"label":'Material'}
            }
        write_property=f'{algorithm_name}-embedding-gc-element-chemenv'

        if use_weights:
            write_property+='-weighted'
        if use_node_properties:
            write_property+='-node_properties'


        return cls(
            name="GC Element ChemEnv",
            node_projections=node_projections,
            relationship_projections=relationship_projections,
            write_property=write_property,
            use_weights=use_weights,
            use_node_properties=use_node_properties
        )

    @classmethod
    def gec_element_chemenv(cls, use_weights=True, use_node_properties=True,algorithm_name='fastRP'):
        relationship_projections={
            "`CHEMENV-GEOMETRIC_ELECTRIC_CONNECTS-CHEMENV`": {"orientation": 'UNDIRECTED', "properties": 'weight'},
            "`ELEMENT-GEOMETRIC_ELECTRIC_CONNECTS-ELEMENT`": {"orientation": 'UNDIRECTED', "properties": 'weight'},
            "`CHEMENV-CAN_OCCUR-ELEMENT`": {"orientation": 'UNDIRECTED', "properties": 'weight'},
            "`MATERIAL-HAS-ELEMENT`": {"orientation": 'UNDIRECTED', "properties": 'weight'},
            "`MATERIAL-HAS-CHEMENV`": {"orientation": 'UNDIRECTED', "properties": 'weight'}
        }
        node_projections={
            "Chemenv": {"label":'Chemenv'},
            "Element": {"label":'Element', 
                        "properties":{
                                'atomic_number':{'default_value':0.0},
                                'X':{'default_value':0.0},
                                'atomic_radius':{'default_value':0.0},
                                'group':{'default_value':0},
                                'row':{'default_value':0},
                                'atomic_mass':{'default_value':0.0}
                                }
                        },
            "Material":{"label":'Material'}
        }
        write_property=f'{algorithm_name}-embedding-ec-element-chemenv'

        if use_weights:
            write_property+='-weighted'
        if use_node_properties:
            write_property+='-node_properties'

        return cls(
            name="EC Element ChemEnv",
            node_projections=node_projections,
            relationship_projections=relationship_projections,
            write_property=write_property,
            use_weights=use_weights,
            use_node_properties=use_node_properties
        )
    
    @classmethod
    def gec_element_chemenv_material_properties(cls, use_weights=True, use_node_properties=True):
        relationship_projections={
                "`CHEMENV-ELECTRIC_CONNECTS-CHEMENV`": {"orientation": 'UNDIRECTED'},
                "`ELEMENT-ELECTRIC_CONNECTS-ELEMENT`": {"orientation": 'UNDIRECTED'},
                "`CHEMENV-CAN_OCCUR-ELEMENT`": {"orientation": 'UNDIRECTED'},
                "`MATERIAL-HAS-ELEMENT`": {"orientation": 'UNDIRECTED'},
                "`MATERIAL-HAS-CHEMENV`": {"orientation": 'UNDIRECTED'}
            }
        node_projections={
                "Chemenv": {"label":'Chemenv'},
                "Element": {"label":'Element' , 
                            "properties":{
                                'atomic_number':{'default_value':0.0},
                                'X':{'default_value':0.0},
                                'atomic_radius':{'default_value':0.0},
                                'group':{'default_value':0.0},
                                'row':{'default_value':0.0},
                                'atomic_mass':{'default_value':0.0}
                                } 
                            },
                "Material":{"label":'Material', "properties":['band_gap','formation_energy_per_atom','energy_per_atom','energy_above_hull','k_vrh','g_vrh']}
            }
        write_property='fastrp-embedding-gec-element-chemenv'

        if use_weights:
            for key, value in relationship_projections.items():
                relationship_projections[key]['properties'] = 'weight'
            write_property+='-weighted'

        if use_node_properties:
            write_property+='-node_properties'

        return cls(
            name="GEC Element ChemEnv",
            node_projections=node_projections,
            relationship_projections=relationship_projections,
            write_property=write_property,
            use_weights=use_weights,
            use_node_properties=use_node_properties
        )

    def get_config(self):
        return {
            "node_projections": self.node_projections,
            "relationship_projections": self.relationship_projections,
            "write_property": self.write_property,
            "use_weights": self.use_weights,
            "use_node_properties": self.use_node_properties
        }


# ['nsites','nelements','volume','density','density_atomic','composition_reduced','formula_pretty'
    #                  'e_electronic','e_ionic','e_total','energy_per_atom','energy_above_hull','formation_energy_per_atom',
    #                  'band_gap','vbm','cbm','efermi',
    #                  'crystal_system','space_group','point_group','hall_symbol'
    #                  'is_gap_direct','is_metal','is_magnetic','is_stable',
    #                  'ordering','total_magnetization','total_magnetization_normalized_vol','num_magnetic_sites','num_unique_magnetic_sites'
    #                  'g_ruess','g_voigt','g_vrh','k_reuss','k_voigt','k_vrh','homogeneous_poisson','universal_anisotropy']

class Neo4jExperimentManager():
    def __init__(self):
        pass
        
    def run_fastRP_algorithm(self,
                            database_name,
                            graph_name,
                            node_projections,
                            relationship_projections,
                            algorithm_params):
        with Neo4jManager() as neo4j_manager:
            manager=Neo4jGDSManager(neo4j_manager)

            try:
                manager.load_graph_into_memory(database_name=database_name,
                                                graph_name=graph_name,
                                                node_projections=node_projections,
                                                relationship_projections=relationship_projections)
            
                print(manager.is_graph_in_memory(database_name,graph_name))
                results=manager.run_fastRP_algorithm(database_name=database_name,
                                            graph_name=graph_name,
                                            **algorithm_params)
                manager.drop_graph(database_name=database_name,graph_name=graph_name)
                print(manager.is_graph_in_memory(database_name,graph_name))
                return results
            except Exception as e:
                print(f"Database {database_name} : {e}")
                pass

    def run_hashGNN_algorithm(self,
                            database_name,
                            graph_name,
                            node_projections,
                            relationship_projections,
                            algorithm_params):
        with Neo4jManager() as neo4j_manager:
            manager=Neo4jGDSManager(neo4j_manager)
            node_property=algorithm_params['mutate_property']
            try:
                manager.load_graph_into_memory(database_name=database_name,
                                                graph_name=graph_name,
                                                node_projections=node_projections,
                                                relationship_projections=relationship_projections)
            
                print(manager.is_graph_in_memory(database_name,graph_name))
                results=manager.run_hashGNN_algorithm(database_name=database_name,
                                            graph_name=graph_name,
                                            **algorithm_params)
                
                manager.write_graph(database_name=database_name,
                                    graph_name=graph_name,
                                    node_properties=node_property,
                                    concurrency=4)
                manager.drop_graph(database_name=database_name,graph_name=graph_name)

                
                print(manager.is_graph_in_memory(database_name,graph_name))
                return results
            except Exception as e:
                print(f"Database {database_name} : {e}")
                pass

    def generate_all_hashGNN(self,
                                     database_names):
        use_weights_list=[True,False]
        use_node_properties_list=[True,False]

        for database_name in database_names:
            LOGGER.info(f"Generating node embeddings for {database_name}")
            

                
            gc_graph_projection=GraphProjection.gc_element_chemenv(use_weights=False,
                                                            use_node_properties=False,
                                                            algorithm_name='hashGNN')
            ec_graph_projection=GraphProjection.ec_element_chemenv(use_weights=False,
                                                            use_node_properties=False,
                                                            algorithm_name='hashGNN')
            gec_graph_projection=GraphProjection.gec_element_chemenv(use_weights=False,
                                                            use_node_properties=False,
                                                            algorithm_name='hashGNN')
            projections=[gc_graph_projection,ec_graph_projection,gec_graph_projection]
            for graph_projection in projections:

                LOGGER.info(f"Using {graph_projection.name}  with {False} weights and {False} node properties")

                relationship_projections=graph_projection.relationship_projections
                node_projections=graph_projection.node_projections
                write_property=graph_projection.write_property

                algorithm_params={'algorithm_mode': 'mutate',
                                'embedding_density':128,
                                'iterations':3,
                                'heterogeneous':True,
                                'mutate_property': write_property}
                # if use_node_properties:
                #     algorithm_params['feature_properties']=graph_projection.node_properties
                #     algorithm_params['binarize_features']={'dimension': 128, 'densityLevel': 2}
                # else:
                algorithm_params['generate_features']={'dimension': 128, 'densityLevel': 2}
        
                for key,value in algorithm_params.items():
                    LOGGER.info(f"Algorithm param: {key} : {value}")

                self.run_hashGNN_algorithm(database_name=database_name,
                                            graph_name='main',
                                            node_projections=node_projections,
                                            relationship_projections=relationship_projections,
                                            algorithm_params=algorithm_params)

                    # self.run_fastRP_algorithm(database_name=database_name,
                    #                                 graph_name='main',
                    #                                 node_projections=node_projections,
                    #                                 relationship_projections=relationship_projections,
                    #                                 algorithm_params=algorithm_params)