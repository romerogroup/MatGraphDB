from math import isclose
from typing import Union

import numpy as np
from coxeter.shapes import ConvexPolygon, ConvexPolyhedron
from mendeleev import element

class UnrecognizedChemicalSpecies(Exception):
    """ Raise this exception if Mendeleev atom species is not correctly assigned."""
    pass


class VoronoiPolyhedron(ConvexPolyhedron):

    """

    A class to represent a Voronoi polyhedron; similar in many respects to
    coxeter's ConvexPolyhedron, and hence inheriting from it, but extended
    in some useful ways for the analysis of Voronoi polyhedra that we will
    need.

    Beyond the functionality already provided by coxeter's ConvexPolyhedron,
    this class provides a chemical species identifying the type of atom that
    the polyhedron is associated with, as well as an ordered list of facets
    in increasing order of their areas. In principle it is possible to have
    facets of the same shape (e.g. triangles) with different areas, so two
    lists are provided: one for the areas and one for the number of vertices
    of the corresponding facet.

    :class: `VoronoiPolyhedron`

    """

    def __init__(self, species: Union[int, str], vertices: np.ndarray, center_atom: np.ndarray = None,) -> None:

        """

        Class Constructor

        :param species: either an int (atomic number) or str (chemical symbol) uniquely
                    identifying the chemical species to which the current Voronoi
                    polyhedron is associated.
        :type Union[int, str]:

        :param vertices: np.ndarray giving the Cartesian coordinates of the vertices
                    defining the Voronoi polyhedron.
        :type np.ndarray:

        :param center_atom: Optional, cartesian coordinates of the center atom in the Voronoi polyhedron
        :type np.ndarray:



        """

        super().__init__(vertices)
        self.species = species
        self.center_atom= center_atom



        try:

            self._species = element(species)

        except:

            raise UnrecognizedChemicalSpecies(species)

        # now loop over faces and calculate their area

        num_corners = np.zeros((self.num_faces), dtype=int)
        areas = np.zeros((self.num_faces), dtype=float)
        face_centers = np.zeros((self.num_faces, 3), dtype=float)
        for nface, face in enumerate(self.faces):
            
            face_corners = vertices[face]

            num_corners[nface] = len(face_corners)
            polygon = ConvexPolygon(face_corners)

            face_centers[nface] = polygon.center

            areas[nface] = polygon.area

        # find the index array that would sort the facet areas

        indices = np.argsort(areas)

        self._polygons = {"area": areas[indices], "shape": num_corners[indices], "face_centers": face_centers[indices]}

        # finally, we will define an array indicating how many
        # polygons of each kind this VP has

        self._num_polygons = np.zeros((np.max(num_corners) - 2), dtype=int)
        # the -2 above is because the minimum number of corners to
        # define a convex polygon is 3

        for nface in range(self.num_faces):

            self._num_polygons[num_corners[nface] - 3] += 1

        self.face_areas = areas

        self.face_centers = face_centers

    def get_polygons(self) -> dict:

        """

        :return: a dictionary containing areas and corners of face polygons
          key: area -> a sorted np array with the areas in increasing order
          key: shape -> the number of corners; shape[i] = number of corners
          for facet with area[i]

        :rtype: dict

        """

        return self._polygons

    def get_num_polygons(self) -> np.ndarray:

        """

        :return: num_polygon array for this VP i.e. the number of faces of each
          type (triangles, squares, etc)

        :rtype: np.ndarray[int]

        """

        return self._num_polygons

    def get_species(self) -> element:

        """

        :return: the mendeleev species associated to this VP

        """

        return self._species

    def get_point_group(self, 
                        area_tolerance:float=0.1,
                        tolerance:float = 0.0001, 
                        eigen_tolerance:float = 0.0001, 
                        matrix_tol:float =0.001) -> str:
        """
        :return: gets and sets the point group attribute a string indicating the point group

        :rtype: str¶
        """
        sa = SymmetryAnalyzerNormals(self ,area_tolerance=area_tolerance, tolerance=tolerance, eigen_tolerance=eigen_tolerance, matrix_tol=matrix_tol)

        self.point_group = sa.point_group
        self.faces_considered = sa.n_faces

        return self.point_group
    
    def get_largest_face_areas(self, tol:float):

        max_area = max(self.face_areas)
        
        largest_face_areas = [ area for area in self.face_areas if area > max_area*(1-tol)]
        return largest_face_areas


    def __eq__( self, other ) -> bool:

        """

        .. document private functions
        .. automethod:: __eq__

        Comparison operator

        For two instances of VoronoiPolyhedron to be considered equal they
        must have the same volume, the same number of faces, the same
        face types, face areas and total face area; otherwise they are
        assumed different. Becase volume and area(s) are floats, we
        base their comparison using intrinsic isclose with a 1.0e-6 tolerance.

        :return: True if self and other are equal accordint to definition above;
          False otherwise

        """

        if not isclose( self.volume, other.volume, abs_tol = 1.0e-6 ):

            return False

        elif not isclose( self.surface_area, other.surface_area, \
                          abs_tol = 1.0e-6 ):

            return False

        elif self.num_vertices != other.num_vertices:

            return False

        elif self.num_faces != other.num_faces:

            return False

        else: # finally check if each face is equal in both instances

            nface = 0

            while nface < self.num_faces:

                if self._polygons['shape'][nface] != \
                       other._polygons['shape'][nface]:

                    return False

                if not isclose( self._polygons['area'][nface], \
                       other._polygons['area'][nface], abs_tol = 1.0e-6 ):

                    return False

                nface += 1

        # if we get this far without return, it has to be because
        # self and other are entirely equivalent

        return True

    def sameshape( self, other ) -> bool:

        """

        Comparison operator for shape

        This is similar to the comparison operator but less stringent; it
        returns True if self and other have the same shape, i.e. same number
        of faces and same face number of corners, but different volume.
        Otherwise returns False.

        :return: True if conditions above are met; False otherwise


        """

        if self.num_vertices != other.num_vertices:

            return False

        elif self.num_faces != other.num_faces:

            return False

        else: # finally check if each face is equal in both instances

            nface = 0

            while nface < self.num_faces:

                if self._polygons['shape'][nface] != \
                     other._polygons['shape'][nface]:

                    return False

                nface += 1

        # if we get this far without return, it has to be because
        # self and other have same shape

        return True
