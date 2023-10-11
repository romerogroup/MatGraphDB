
import os
from typing import List, Tuple, Union

import numpy as np
from scipy.spatial import Voronoi, distance

import pymatgen.core as pmat

from poly_graphs_lib.core.structure import Structure
from poly_graphs_lib.utils import periodic_table
from poly_graphs_lib.core.voronoi_polyhedron import VoronoiPolyhedron


class VoronoiStructure(Structure):
    """
    A class to represent the full voronoi analysis of a structure. 
    It should naturally extend from Structure since it should still desribe 
    the structure, but it will also have additional attributes and methods that 
    will add us in databasing and the Voronoi Analysis.   

    
    :class: `VoronoiPolyhedron`
    """
    def __init__(self,structure_id: Union[int, str], 
                    database_source:str=None,
                    database_id:str=None,
                    neighbor_tol:float=0.05):

        super().__init__(structure_id=structure_id)
        
        r"""

        Class Constructor

        :param structure_id: the identification method of a structure this will either be an
                    int corresponding to the desired entry in COD or 
                    the (path)filename of a local cif file.
        :type Union[int, str]:

        :param database_source: Optional, str that will identify the source database
                            i.e. ['mp','cod','jarvis',etc..]
        :type str:

        :param database_id: Optional, the corresponding id in the source database
        :type str:

        :param neighbor_tol: Optional, tolerance for to count bonds.
        :type float:

        """
        self.database_source = database_source
        self.database_id = database_id
        self.neighbor_tol=neighbor_tol

        self._chemenv_analysis()
        self._voronoi_analysis()

    def as_dict(self):
        """

        :return: a dictionary describing fully describing the structure
        :rtype: dict

        """

        tmp_dict = {
                    'database_id':self.database_id,
                    'database_source':self.database_source,
                    'formula':self.formula,
                    'composition':self.composition,
                    'lattice':self.direct_lattice.tolist(),
                    'frac_coords':self.frac_coords_unit.tolist(),
                    'atoms':self.atoms_unit,
                    'atom_density':self.atom_density,
                    'mass_density':self.mass_density,
                    'unit_cell_volume':self.volume,
                    'spg_number':self.spg_number,
                    'crytstal_system':self.crystal_system,
                    'e_above_hull':self.e_above_hull,
                    'combined_unique_voronoi_volume':self.combined_voronoi_volume,
                    'combined_unique_voronoi_area':self.combined_voronoi_surface_area,
                    'voronoi_polyhedra_info':self.voronoi_polyhedra_dicts
                    }
        return tmp_dict

    def _voronoi_analysis(self):
        """
        Helper function to perform the voronoi analysis

        :return: None
        :rtype: None
        """
        atoms, report_voronoi,report_voronoi_unit = self.get_atom_lists()

        indices, = np.nonzero(report_voronoi)
        indices_unit, = np.nonzero(report_voronoi_unit)

        # find voronoi polyhedra for this structure
        voronoi = Voronoi( self.cart_coords)
        self.voronoi_polyhedra = []
        self.voronoi_polyhedra_dicts = []
        self.combined_voronoi_volume=0
        self.combined_voronoi_surface_area = 0

        for ipoly in indices_unit:

            species = self.atoms[ipoly]


            unit_index= self.unit_index_map[ipoly]
            # Find the voronoi region of the 
            target_region=voronoi.point_region[ipoly]
            neighbors=[]
            neighbors_unit=[]
            for p1, p2 in voronoi.ridge_points:
                if p1 == target_region:
                    neighbors.append(voronoi.point_region[p2])

                    neighbors_unit.append(self.unit_index_map[p2])
                elif p2 == target_region:
                    neighbors.append(voronoi.point_region[p1])

                    neighbors_unit.append(self.unit_index_map[p1])
            neighbor_coordination_envrionment=[self.coordination_environments[index] for index in neighbors_unit]
            
            vertices = voronoi.vertices[voronoi.regions[voronoi.point_region[ipoly]]]
            center_atom = self.cart_coords[ipoly,:]

            # Voronoi Analysis
            voronoi_polyhedron = VoronoiPolyhedron( species=species, 
                                                    vertices = vertices, 
                                                    center_atom = center_atom)
  
            self.combined_voronoi_volume += voronoi_polyhedron.volume
            self.combined_voronoi_surface_area += voronoi_polyhedron.surface_area
            # Determination of the corrdination number
            covalent_radii_atoms = np.array([periodic_table.covalent_radii[atomic_number] for atomic_number in self.atoms])
            covelent_radius_center = periodic_table.covalent_radii[self.atoms[ipoly]]
            
            # Calculates the proposed bond length based on the covalent radii and a tolerance
            proposed_bond_distances = (covalent_radii_atoms + np.array([covelent_radius_center]*len(self.atoms)) ) * (1 + self.neighbor_tol)

            # Calculates the actual distance between center atom and the ther sites
            distance_array = np.linalg.norm(voronoi_polyhedron.center_atom - self.cart_coords, axis=1)
            
            # Determines which sites are the closet to the center site and exludes the center site itself
            bond_indices = np.where(np.logical_and(distance_array < proposed_bond_distances, distance_array != 0))[0]
            
            # Some center site properties
            center_coordination = bond_indices.shape[0]
            center_block = periodic_table.blocks[self.atoms[ipoly]]
            center_period = periodic_table.periods[self.atoms[ipoly]]
            center_group = periodic_table.groups[self.atoms[ipoly]]
            center_atomic_number = self.atoms[ipoly]

            
            # Storing information in a jsonable dictionary
            polyhedron_dict = {'vertices': voronoi_polyhedron.vertices.tolist(),
                               'neighbor_unit_indices':neighbors_unit,
                                'species': self.atoms[ipoly],
                                'coordination_envrionment':self.coordination_environments[unit_index],
                                'neighbor_coordination_envrionment':neighbor_coordination_envrionment,
                                'center_atom': voronoi_polyhedron.center_atom.tolist(),
                                'center_atom': voronoi_polyhedron.center_atom.tolist(),
                                'voronoi_volume':voronoi_polyhedron.volume,
                                'voronoi_surface_area':voronoi_polyhedron.surface_area,
                                'voronoi_face_areas':voronoi_polyhedron.face_areas.tolist(),
                                'center_coordination':center_coordination,
                                'center_block':center_block,
                                'center_period':center_period,
                                'center_groups':center_group,
                                'center_atomic_number':center_atomic_number,
                                }

            self.voronoi_polyhedra_dicts.append(polyhedron_dict)
            self.voronoi_polyhedra.append(voronoi_polyhedron)

    def _chemenv_analysis(self):
        coords =  self.frac_coords_unit
        lattice = self.direct_lattice
        atoms=self.atoms_unit
        struct = pmat.Structure(lattice=lattice, 
                            species=atoms, 
                            coords=coords)

        lgf = LocalGeometryFinder()

        #you can also save the logging to a file, just remove the comment
        # logging.basicConfig(#filename='chemenv_structure_environments.log',
        #                     format='%(levelname)s:%(module)s:%(funcName)s:%(message)s',
        #                     level=logging.DEBUG)
        lgf.setup_structure(structure=struct)

        se = lgf.compute_structure_environments(maximum_distance_factor=1.41,only_cations=False)
        strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3)
        lse = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=se)
        # list of possible coordination environements per site
        self.coordination_environments = lse.coordination_environments