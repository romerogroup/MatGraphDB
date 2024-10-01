from matgraphdb._version import __version__

# from matgraphdb.data.manager import DBManager
# from matgraphdb.graph.neo4j_gds_manager import Neo4jGDSManager
# from matgraphdb.graph.neo4j_manager import Neo4jManager
# from matgraphdb.graph.graph_generator import GraphGenerator

from matgraphdb.core import MatGraphDB
from . import _version
__version__ = _version.get_versions()['version']
