import os


from poly_graphs_lib.utils import test_polys,test_names
os.environ['KMP_DUPLICATE_LIB_OK']='True'


verts_tetra = PlatonicFamily.get_shape("Tetrahedron").vertices
verts_cube = PlatonicFamily.get_shape("Cube").vertices
verts_oct = PlatonicFamily.get_shape("Octahedron").vertices
verts_dod = PlatonicFamily.get_shape("Dodecahedron").vertices

# verts_tetra_rot, verts_tetra
verts_tetra_scaled  = 2*verts_tetra

# verts = test_polys[-3]1
verts = verts_tetra
# verts = verts_oct
# print(test_names)
print(verts_tetra)


# verts_list = [verts_tetra,verts_cube,verts_oct,verts_dod]
# verts_labels = ['tetra','cube','oct','dod']

