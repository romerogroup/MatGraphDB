import random

from coxeter.families import PlatonicFamily
from coxeter.shapes import ConvexPolyhedron
from scipy.spatial import ConvexHull
import numpy as np
import pyvista as pv
# class PolyhedronGenerator:
#     """
#     This class is used to generate augmentations of a polyhedron
#     """

#     def __init__(self):
#         pass

#     def __call__(self,verts):
#         self.verts = verts

#     def random_vertex_translation(self):
#         n_verts = len(self.verts )

#         i_vert = random.randint(0,n_verts-1)

#         verts = self.verts.copy()
#         verts[i_vert,:] = self.verts[i_vert,:] + 0.01

#         if self.is_valid_augmentation():
#             return

#     def is_valid_augmentation(self):
#         is_valid = True
#         try:
#             ConvexPolyhedron()
#         except:
#             is_valid = False
#         return is_valid

def is_valid_augmentation(verts):
    is_valid = True
    try:
        ConvexPolyhedron(verts)
    except:
        is_valid = False
    return is_valid

def random_vertex_translation(verts, random_range=[-0.1,0.1] ):

    n_verts = len(verts)

    i_vert = random.randint(0,n_verts-1)

    found_valid = False
    while found_valid == False:
        verts = verts.copy()
        
        for i_coord in range(3):
            translation_value = random.uniform(*random_range)
            verts[i_vert,i_coord] = verts[i_vert,i_coord] + translation_value

        if is_valid_augmentation(verts = verts):
            found_valid= True
    return verts

def perpendicular_vector(v):
    """
    Generates a vector perpendicular to a given vector.
    
    Args:
    v (np.ndarray): A 1D array of length 3 representing a vector in 3D space.
    
    Returns:
    np.ndarray: A 1D array of length 3 representing a vector perpendicular to v.
    """
    if np.array_equal(v, np.array([0, 0, 0])):
        raise ValueError("Input vector cannot be the zero vector.")
    if np.array_equal(v, np.array([1, 0, 0])):
        return np.array([0, 1, 0])
    else:
        return np.cross(v, np.array([1, 0, 0]))
    
# def replace_vertex(points, vertex_idx, tolerance):
#     # Get the original vertex and its neighbors
#     vertex = points[vertex_idx]
#     com = np.mean(points, axis=0)


#     vertex_direction = vertex - com

#     perp_vector = perpendicular_vector(v=vertex_direction)

#     new_point

    

def add_polyhedra(plotter,verts):
    plotter.add_mesh(verts, render_points_as_spheres=True, point_size=15, color='blue')
    plotter.add_mesh(pv.PolyData(verts).delaunay_3d(), color='green')
    return None

def main():

    verts_tetra = PlatonicFamily.get_shape("Tetrahedron").vertices
    verts_cube = PlatonicFamily.get_shape("Cube").vertices
    verts_oct = PlatonicFamily.get_shape("Octahedron").vertices
    verts_dod = PlatonicFamily.get_shape("Dodecahedron").vertices

    plotter = pv.Plotter()

    add_polyhedra(plotter,verts=verts_tetra)
    # add_polyhedra(plotter,verts= random_vertex_translation(verts_tetra, random_range=[-0.1,0.1] ))
    # add_polyhedra(plotter,verts= replace_vertex(points=verts_tetra, vertex_idx=0,tolerance=0.1))


    plotter.show()
    
    

if __name__=="__main__":
    main()