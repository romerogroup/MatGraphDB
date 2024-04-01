from os.path import exists
from typing import List, Tuple, Union
import json

import numpy as np
from crystals import Crystal
from pymatgen.core import Structure
import spglib

from matgraphdb.utils import periodic_table

class CODidNotFound(Exception):
    """Raise this exception if COD database ID is incorrect or not found"""
    pass

class Structure:

    """

    This class represents a crystal structure that is either read from a cif file
    or from an appropriate database (e.g. Crystallographic Open Database, COD). There are
    other tools that do this, such as crystals or the spglib module, but what we
    want here is something more specifically directed at calculating Voronoi
    polyhedra. In order to do this, the present class, after initialization and
    construction of the required structure, will perform the following functions:

    1) it will generate a 3x3x3 supercell of the primitive cell of the structure so
    that the central unit cell is totally surrounded by periodic images of itself;
    this ensures that all nearest neighbours of the atoms in the central unit cell
    are considered.

    2) crystallographic facilities from Crystal will be employed to identify a unique
    representative atom for each crystallographic orbit in the crystal, so that
    Voronoi polyhedra are only calculated later for the set of unique atoms of each
    chemical species within the structure.

    """

    def __init__(self, structure_id: Union[int, str]=None, pmat_structure_file=None) -> None:

        r"""

        Create an instance of Structure; the constructor accepts either an int corresponding
        to the desired entry in COD or the (path)filename of a local cif file.

        :param structure_id: the identification method of a structure this will either be an
                    int corresponding to the desired entry in COD or 
                    the (path)filename of a local cif file.
        :type Union[int, str]:

        """

        if isinstance(structure_id, str):  # from a file

            if not exists(structure_id):

                raise FileNotFoundError(structure_id)

            unit_cell = Crystal.from_cif(structure_id)

        elif isinstance(structure_id, int):

            try:

                unit_cell = Crystal.from_cod(structure_id)

            except:

                raise CODidNotFound(structure_id)
            
        # elif:
        #     with open(pmat_structure_file) as file:
        #         dct = json.load(file)
        #         pmat_structure = Structure.from_dict(dct)
            
            # lattice_vectors=pmat_structure.lattice
            # unitcell=
            # unit_cell = Crystal(unitcell, lattice_vectors)


        # we will try to find a primitive cell; this need not always work; when
        # it doesn't we'll just use the cell as read in
        
        try:
            self.primitive = unit_cell.primitive()
        except:
            self.primitive = unit_cell

        # properties belonging to the unitcell
        
        self.frac_coords_unit = np.array([atom.coords_fractional for atom in self.primitive])
        self.cart_coords_unit = np.array([atom.coords_cartesian for atom in self.primitive])
        self.atoms_unit = [atom.atomic_number for atom in self.primitive]
        self.atoms_masses_unit = [periodic_table.masses[atom.atomic_number] for atom in self.primitive]
        self.atoms_mass_density_unit = self.atoms_masses_unit/self.volume
        self.atoms_atom_density_unit = len(self.atoms_unit)/self.volume

        sorted_indices = np.lexsort((self.frac_coords_unit[:, 2], self.frac_coords_unit[:, 1], self.frac_coords_unit[:, 0]))
        self.frac_coords_unit=self.frac_coords_unit[sorted_indices]
        self.cart_coords_unit=self.cart_coords_unit[sorted_indices]
        self.atoms_unit=[self.atoms_unit[i] for i in sorted_indices]
        self.atoms_masses_unit=[self.atoms_masses_unit[i] for i in sorted_indices]
        self.atoms_mass_density_unit=[self.atoms_mass_density_unit[i] for i in sorted_indices]
    
        # now generate a 3x3x3 supercell in which the basic cell
        # (the one read in) is the central one

        self.n_atoms = 27 * len(self.primitive)

        self.coordinates = np.zeros((self.n_atoms, 3), dtype=float)  # fractional coords
        self.voronoi = np.zeros((self.n_atoms), dtype=bool)

        self.voronoi_unit = np.zeros((self.n_atoms), dtype=bool)
        self.unit_index_map=np.zeros((self.n_atoms), dtype=int)
        self.atoms = []

        # we will separate atoms in groups according to which crystallographic orbit
        # they belong to; we only need to calculate Voronoi polyhedra for one atom
        # in each orbit; the remaining atoms in the orbit are equivalent by symmetry

        self.orbits = self.primitive.groupby("crystallographic_orbits")
        i = 0
        # note: orbit variable is currently unused, but it may be worthwhile using
        # it to identify each atom as pertaining to a specific crystal orbit
        for n_c in range(-1, 2):
            for n_b in range(-1, 2):
                for n_a in range(-1, 2):
                    for orbit, atoms in self.orbits.items():
                        first = 0
                        for i_atom,atom in enumerate(atoms):
                            
                            self.coordinates[i,:] = \
                               atom.coords_fractional + np.array([n_a, n_b, n_c])

                            # Voronoi by grouping
                            if n_a == n_b == n_c == 0 and first == 0:
                                self.voronoi[i] = True
                            
                            # Voronoi unit cell
                            if n_a == n_b == n_c == 0:
                                self.voronoi_unit[i]=True
                            self.atoms.append(atom.atomic_number)



                            diff = np.abs(self.frac_coords_unit - atom.coords_fractional)
                            # Check if any row in diff is close to zero (within the specified tolerance) along all columns
                            match_row = np.all(diff < 1e-6, axis=1)
                            unit_cell_index = np.where(match_row)[0][0]
                            self.unit_index_map[i]=unit_cell_index

                            i += 1
                            first += 1

        # Extra properties

        # This property can be obtained from materials project
        self.e_above_hull = None

    def get_n_atoms( self ) -> int:

        """

        :return: number of atoms in the 3x3x3 supercell.
        :rtype: int

        """

        return self.n_atoms

    def get_n_atoms_unit_cell( self ) -> int:

        """

        :return: number of atoms in the primitive cell
        :rtype: int

        """

        return len(self.primitive)

    def get_atom_lists( self ) -> Tuple[List, np.ndarray]:

        """

        Returns the list of atoms species (atomic_numbers) and a bool array; the latter
        is set to False in general and True only for a representative
        atom of each crystallographic orbit in the central unit cell; Voronoi polyhedra
        will be reported only for atoms for which this flag is set to True

        :return: list of atomic numbers for all atoms in the 3x3x3 supercell
        :rtype: List[int]
        :return: bool array for reported Voronoi polyhedra
        :rtype: np.ndarray[bool]

        """

        return self.atoms, self.voronoi,self.voronoi_unit

    def get_lattice_vectors( self ) -> np.ndarray:

        """

        :return: the lattice vectors in Angstrom, as a 3x3 np.ndarray, a vector per array row.
        :rtype: np.ndarray[3,3]

        """

        direct_lattice = np.array(self.primitive.lattice_vectors)
        return direct_lattice

    def get_reciprocal_vectors( self ) -> np.ndarray:

        """

        :return: the reciprocal lattice vectors as a 3x3 np.ndarray, a vector per array row.
        :rtype: np.ndarray[3,3]

        """

        rec_lattice = np.array(self.primitive.reciprocal_vectors)
        return rec_lattice

    def get_lattice_parameters( self ) -> Tuple:

        """

        :return: the lattice parameters in Angstrom, angles in degrees a, b, c, alpha, beta, gamma
        :rtype: Tuple[float, float, float, float, float, float]

        """

        return self.primitive.lattice_parameters

    def get_fractional_coordinates( self ) -> np.ndarray:

        """

        Since the supercell is constructed such that the central unit cell has
        lattice coordinates in [0,1], some of the coordinates for other cells will
        be negative, while some of the coordinates will be greater than 1.

        :return: a numpy array of lattice coordinates of the supercell (n_atoms,3);
        :rtype: np.ndarray

        """

        return self.coordinates

    def get_cartesian_coordinates( self ) -> np.ndarray:

        """

        :return: a numpy array of cartesian coordinates in Angstrom (n_atoms,3)
        :rtype: np.ndarray

        """

        direct_lattice = np.array(self.primitive.lattice_vectors)
        # For some reason the z and x axis were reversed
        # direct_lattice[:,[2,0]] = direct_lattice[:,[0,2]]
        # This is equivalent to the below for loop
        cartesian_coordinates = self.coordinates.dot(direct_lattice)
        #
        # for i in range(self.n_atoms):

        #     cartesian_coordinates[i, :] = np.dot(
        #         direct_lattice.T , self.coordinates[i, :]
        #     )


        # for icoord, coord in enumerate(self.coordinates):
        #     print(icoord , self.coordinates[icoord,:], cart_coords[icoord,:])
        # print(np.round_(np.array([[0,0,0],[0.25,0.25,0.25],[0.75,0.75,0.75]])\
        # .dot(direct_lattice), decimals = 3, out = None))

        return cartesian_coordinates

    # # Logan : This is if we want to avoid pymatgen
    # @property
    # def _spglib_cell(self):
    #     return (self.direct_lattice,  self.atoms_unit, self.frac_coords_unit)

    # def get_wyckoff_positions(self, symprec=1e-5):
    #     n_atoms =  len(self.atoms_unit)
    #     wyckoff_positions = np.empty(shape=(n_atoms), dtype="<U4")
    #     print(self._spglib_cell)
    #     print(spglib.get_symmetry_dataset(self._spglib_cell, symprec))
    #     wyckoffs_temp = np.array(
    #         spglib.get_symmetry_dataset(self._spglib_cell, symprec)["wyckoffs"]
    #     )
    #     group = np.zeros(shape=(n_atoms), dtype=np.int)
    #     counter = 0
    #     for iwyckoff in np.unique(wyckoffs_temp):
    #         idx = np.where(wyckoffs_temp == iwyckoff)[0]
    #         for ispc in np.unique(self.atoms[idx]):
    #             idx2 = np.where(self.atoms[idx] == ispc)[0]
    #             multiplicity = len(idx2)
    #             wyckoff_positions[idx][idx2]
    #             for i in idx[idx2]:
    #                 wyckoff_positions[i] = str(multiplicity) + iwyckoff
    #                 group[i] = counter
    #             counter += 1
    #     self.wyckoff_positions = wyckoff_positions
    #     self.group = group
    #     return wyckoff_positions

    def get_valences(self):

        import pymatgen.core as pmat
        from pymatgen.analysis.bond_valence import BVAnalyzer


        s = pmat.Structure(lattice = self.direct_lattice, 
                            species = self.atoms_unit, 
                            coords = self.frac_coords_unit)


        bva = BVAnalyzer()
        self.valences_unit = bva.get_valences(structure = s)

        atom_to_valence_mapping = {Z:valence for Z, valence in zip(self.atoms_unit,self.valences_unit)}

        # Map unit cell valences to supercell. 
        # Logan : I believe this order of the wycoff
        self.valences = [] 


        for n_c in range(-1, 2):
            for n_b in range(-1, 2):
                for n_a in range(-1, 2):
                    for orbit, atoms in self.orbits.items():
                        for atom in atoms:
                            for frac_coord_unit,valence_unit in zip(self.frac_coords_unit,self.valences_unit):
                                if np.array_equal(atom.coords_fractional, frac_coord_unit):
                                    self.valences.append(valence_unit)

        # for atom, valence in zip(self.atoms,self.valences):
        #     print(atom, valence)

        return self.valences

    def get_wyckoff_positions(self):
        from pyxtal.symmetry import Group
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        import pymatgen.core as pmat

        s = pmat.Structure(lattice = self.direct_lattice, 
                            species = self.atoms_unit, 
                            coords = self.frac_coords_unit)
        
        sa = SpacegroupAnalyzer(structure=s, symprec=0.01, angle_tolerance=5.0)
        s_dict = sa.get_symmetry_dataset()

        # print(s_dict)
        self.wyckoffs_letter_unit = np.array(s_dict['wyckoffs'])

        # print(len(self.wyckoffs_letter_unit ))
        self.wyckoff_positions_unit = ['']*len(self.wyckoffs_letter_unit)
        for iwyckoff in np.unique(self.wyckoffs_letter_unit):
            idx = np.where(self.wyckoffs_letter_unit == iwyckoff)[0]
            for ispc in np.unique(np.array(self.atoms_unit)[idx]):
                idx2 = np.where(np.array(self.atoms_unit)[idx] == ispc)[0]
                multiplicity = len(idx2)
                for i in idx[idx2]:
                    self.wyckoff_positions_unit[i] = str(multiplicity) + iwyckoff

        # print(self.atoms_unit)
        # print( self.wyckoff_positions_unit)
        # print(s_dict['wyckoffs'])
        # print(s_dict['site_symmetry_symbols'])
        # print(s_dict['std_types'])

        # Map unit cel l wycoff positions to supercell. 
        # Logan : I believe this order of the wycoff
        self.wyckoffs = []
        for n_c in range(-1, 2):
            for n_b in range(-1, 2):
                for n_a in range(-1, 2):
                    for orbit, atoms in self.orbits.items():
                        for atom in atoms:
                            for frac_coord_unit,wyckoff_unit in zip(self.frac_coords_unit,self.wyckoff_positions_unit):
                                if np.array_equal(atom.coords_fractional, frac_coord_unit):
                                    self.wyckoffs.append(wyckoff_unit)


        return self.wyckoffs

    def get_crystal_system(self):
        """

        :return: a string of what crystal system 
        ['triclinic','monoclinic','orthorhombic','tetragonal','trigonal','hexagonal','cubic']
        :rtype: str

        """
        crystal_system = None
        if self.spg_number <= 2:
            crystal_system = 'triclinic'
        elif self.spg_number <= 15:
            crystal_system = 'monoclinic'
        elif self.spg_number <= 74:
            crystal_system = 'orthorhombic'
        elif self.spg_number <= 142:
            crystal_system = 'tetragonal'
        elif self.spg_number <= 167:
            crystal_system = 'trigonal'
        elif self.spg_number <= 194:
            crystal_system = 'hexagonal'
        else:
            crystal_system = 'cubic'

        return crystal_system

    def set_e_above_hull(self, e_above_hull: float):
        """
        Sets e_above_hull if available. Materials project has this property.
        :return: None
        :rtype: None

        """
        self.e_above_hull = e_above_hull
        return None
    
    @property
    def composition(self):
        "Sets fcomposition as an attribute"
        return self.primitive.chemical_composition

    @property
    def volume(self):
        "Sets volume as an attribute"
        return self.primitive.volume

    @property
    def mass_density(self):
        "Sets density as an attribute"
        return np.sum(self.atoms_mass_density_unit)

    @property
    def atom_density(self):
        "Sets density as an attribute"
        return self.atoms_atom_density_unit

    @property
    def formula(self):
        "Sets spg_number as an attribute"
        return self.primitive.chemical_formula

    @property
    def spg_number(self):
        "Sets spg_number as an attribute"
        return self.primitive.international_number

    @property
    def crystal_system(self):
        "Sets crystal_system as an attribute"
        return self.get_crystal_system()

    @property
    def direct_lattice(self):
        "Sets lattice as an attribute"
        return self.get_lattice_vectors()

    @property
    def reciprocal_lattice(self):
        "Sets reciprocal lattice as an attribute"
        return self.get_reciprocal_vectors()

    @property
    def cart_coords(self):
        "Sets cartesian coordinates as an attribute"
        return self.get_cartesian_coordinates()

    @property
    def frac_coords(self):
        "Sets fraction coordinates as an attribute"
        return self.get_fractional_coordinates()

    @property
    def lattice_parameters(self):
        "Sets lattice parameters as an attribute"
        return self.get_lattice_parameters()
