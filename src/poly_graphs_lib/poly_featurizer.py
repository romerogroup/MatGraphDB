import json
import numpy as np
from coxeter.families import PlatonicFamily
from coxeter.shapes import ConvexPolyhedron
from torch.nn import functional as F
# import encoder


from voronoi_statistics.similarity_measure_random import similarity_measure_random

class PolyFeaturizer:
    
    def __init__(self, vertices, norm:bool=False):

        self.poly = ConvexPolyhedron(vertices=vertices)
        self.poly.merge_faces(atol=1e-4, rtol=1e-5)

        self.norm = norm

        self.x = None
        self.edge_index = None
        self.edge_attr = None
        self.y = None
        self.pos = None

        self.initialize_properties()

    def initialize_properties(self):

        if self.norm:
            self.volume = self.poly.volume
            self.vertices = self.get_vertices()
            vertices = self.normalize(self.vertices, volume = self.volume)

            self.poly = ConvexPolyhedron(vertices=vertices)
            self.poly.merge_faces(atol=1e-4, rtol=1e-5)

        self.volume = self.poly.volume
        self.face_centers = self.get_face_centers()
        self.face_normals = self.get_face_normals()
        self.vertices = self.get_vertices()

        self.faces = self.get_faces()
        self.face_sides = self.get_face_sides()
        self.edges = self.get_edges()
        self.edge_centers = self.get_edge_centers()

        


        self.verts_adj_mat = self.get_vertices_adjacency()
        self.faces_adj_mat = self.get_faces_adjacency()
        self.face_edges = self.get_face_edges()
        self.face_areas = self.get_face_areas()

        return None

    def normalize(self,points , volume):
        points = (points - points.mean(axis=0))/volume**(1/3)

        # points = (points - points.mean(axis=0))/points.std(axis=0)
        return points
    
    def get_face_centers(self):
        face_centers=[]
        for face in self.poly.faces:
            face_center = np.mean(self.poly.vertices[face], axis=0)
            face_centers.append(face_center)
        return np.array(face_centers)
    
    def get_edge_centers(self):
        edge_centers=[]
        for edge in self.edges:
            edge_center = np.mean(self.vertices[edge,:], axis=0)
            edge_centers.append(edge_center)
        return np.array(edge_centers)
    
    def get_face_normals(self):
        return self.poly.normals

    def get_vertices(self):
        return self.poly.vertices
    
    def get_faces(self):
        return self.poly.faces

    def get_face_sides(self):
        face_sides=[]
        for face in self.faces:
            face_sides.append(len(face))
        return np.array(face_sides)

    def get_edges(self):
        edges = set()
        for face in self.faces:
            for i in range(len(face)):
                # Each edge is defined by a pair of consecutive vertices in the face.
                edge = (face[i], face[(i+1)%len(face)] )  
                # We always store the smaller vertex index first.
                if edge[0] > edge[1]:
                    edge = (edge[1], edge[0]) 
                edges.add(edge)
        edges = sorted(list(edges))
        return edges
    
    def get_vertices_adjacency(self):
        n = len(self.vertices)
        adj_matrix = np.zeros((n, n), dtype=int)
        for edge in self.edges:
            adj_matrix[edge[0], edge[1]] = 1
            adj_matrix[edge[1], edge[0]] = 1
        return adj_matrix
    
    def get_faces_adjacency(self):
        n_faces = len(self.faces)
        adj_matrix = np.zeros((n_faces, n_faces), dtype=int)
        for i in range(n_faces):
            for j in range(i+1, n_faces):
                if any((v in self.faces[i]) and (v in self.faces[j]) for v in range(len(self.vertices))):
                    adj_matrix[i, j] = True
                    adj_matrix[j, i] = True
        return adj_matrix

    def get_face_edges(self):
        n_faces = len(self.faces_adj_mat)
        edges=set()
        for i in range(n_faces):
            for j in range(i+1,n_faces):
                if self.faces_adj_mat[i,j] > 0:
                    edge = (i,j)

                    if edge[0] > edge[1]:
                        edge = (edge[1], edge[0]) 
                    edges.add(edge)
        edges = sorted(list(edges))

        return np.array(edges)

    def get_face_areas(self):
        face_areas=[]
        for i,face in enumerate(self.faces):
            face_area=self.poly.get_face_area(i)
            face_areas.append(face_area)
        return np.array(face_areas)
        
    def get_dihedral_angles(self):
        n_edges = len(self.face_edges)
        dihedral_angles = np.zeros(shape = (n_edges))
        for index, edge in enumerate(self.face_edges):
            i = edge[0]
            j = edge[1]
            dihedral_angles[index] = np.arccos(np.dot(-self.face_normals[i], self.face_normals[j]))
        return dihedral_angles.reshape(-1,1)
    
    def get_edge_lengths(self):
        n_edges = len(self.edges)
        edge_lengths = np.zeros(shape = (n_edges))
        for index, edge in enumerate(self.edges):
            i = edge[0]
            j = edge[1]
            edge_lengths[index] = np.linalg.norm(self.vertices[i,:]-self.vertices[j,:])
        return edge_lengths

    def get_energy_per_node(self, nodes, adj_mat):
        n_nodes = len(nodes)
        energy = 0
        non_zero_edges = 0
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if adj_mat[i,j] > 0:
                    distance = np.linalg.norm(nodes[i] - nodes[j])
                    energy += distance
                    non_zero_edges +=1
        return energy / non_zero_edges

    def get_three_body_energy(self, nodes, face_normals, adj_mat):
        energy = 0
        n_nodes = len(nodes)
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if adj_mat[i,j] > 0:
                    sep_vec = nodes[i] - nodes[j]

                    dot_ij = np.dot(-face_normals[i], face_normals[j])
                    if abs(dot_ij) > 1:
                        dot_ij =1
                    dihedral_angle = np.arccos(dot_ij)
                    
                    energy += 0.5*np.linalg.norm(sep_vec)**2 + 0.5 * dihedral_angle**2
        return energy
     
    def get_moment_of_inertia(self, nodes):
        n_nodes = len(nodes)
    
        nodes = (nodes - nodes.mean(axis=0))/nodes.std(axis=0)
        center_of_mass = nodes.mean(axis=0)
        distances_squared = np.sum((nodes - center_of_mass)**2, axis=1)
        return float(distances_squared.sum())/ n_nodes
    
    def compare_poly(self, vertices , ncores=1,n_points=10000):
        
        similarity = similarity_measure_random(verts_a =self.vertices,
                                                verts_b=vertices , 
                                                n_cores=ncores,
                                                n_points=n_points, 
                                                sigma=2)
        return similarity
        
    def get_pyg_faces_input(self, x=None, edge_attr=None, y=None):
        edge_index = self.face_edges
        pos = self.face_centers

        if x is None:
            x = np.arange(len(self.face_centers)).reshape(-1,1)

        if y is None:
            y = 1

        self.x=x
        self.edge_index=edge_index.T
        self.edge_attr=edge_attr 
        self.y=y 
        self.pos = pos
        return x, edge_index, edge_attr, y, pos
    
    def get_pyg_verts_input(self, x=None, edge_attr=None, y=None):
        edge_index = self.edges
        pos = self.vertices

        if x is None:
            x = np.arange(len(self.vertices)).reshape(-1,1)

        if y is None:
            y = 1

        self.x=x
        self.edge_index=edge_index.T
        self.edge_attr=edge_attr 
        self.y=y 
        self.pos = pos

        return x, edge_index, edge_attr, y, pos
    
    def set_label(self, label):
        self.label = label

        return None
    
    def export_pyg_json(self, filename:str = None):
        data = {
                'x':self.x.tolist(),
                'edge_index':self.edge_index.tolist(),
                'edge_attr':self.edge_attr.tolist(),
                'pos':self.pos.tolist(),
                'y':self.y,
                'label':self.label
                }
        
        if filename:
            with open(filename,'w') as outfile:
                json.dump(data, outfile,indent=4)

        return data
    

if __name__ == "__main__":

    verts_tetra = PlatonicFamily.get_shape("Tetrahedron").vertices
    verts_tetra = PlatonicFamily.get_shape("Cube").vertices
    verts_oct = PlatonicFamily.get_shape("Octahedron").vertices
    verts_dod = PlatonicFamily.get_shape("Dodecahedron").vertices

    # verts_tetra_rot, verts_tetra
    verts_tetra_scaled  = 2*verts_tetra
    # verts_tetra_rot  = (verts_tetra_rot - verts_tetra_rot.mean(axis=0))/verts_tetra_rot.std(axis=0)
    # verts_tetra  = (verts_tetra - verts_tetra.mean(axis=0))/verts_tetra.std(axis=0)

    # poly_rotated = ConvexPolyhedron(vertices=verts_tetra_rot)
    # poly_tetra = ConvexPolyhedron(vertices=verts_tetra)


    # verts_tetra_rot  = (verts_tetra_rot - verts_tetra_rot.mean(axis=0))/verts_tetra_rot.std(axis=0)
    # verts_tetra  = (verts_tetra - verts_tetra.mean(axis=0))/verts_tetra.std(axis=0)
    
    # # verts_tetra_rot  = (verts_tetra_rot - verts_tetra_rot.min(axis=0))/(verts_tetra_rot.max(axis=0) - verts_tetra_rot.min(axis=0))
    # # verts_tetra  = (verts_tetra - verts_tetra.min(axis=0))/(verts_tetra.max(axis=0) - verts_tetra.min(axis=0))
    # obj = PolyFeaturizer(vertices=verts_tetra)
    # face_sides_features = encoder.face_sides_bin_encoder(obj.face_sides)
    # face_areas_features = obj.face_areas
    # node_features = np.concatenate([face_areas_features,face_sides_features],axis=1)

    # target_variable = obj.get_three_body_energy(nodes=obj.face_centers,
    #                                                 face_normals=obj.face_normals,
    #                                                 adj_mat=obj.faces_adj_mat)
    # print(target_variable)
    # print(node_features)


    # obj = PolyFeaturizer(vertices=verts_tetra_scaled)
    # face_sides_features = encoder.face_sides_bin_encoder(obj.face_sides)
    # face_areas_features = obj.face_areas
    # node_features = np.concatenate([face_areas_features,face_sides_features],axis=1)

    # target_variable = obj.get_three_body_energy(nodes=obj.face_centers,
    #                                                 face_normals=obj.face_normals,
    #                                                 adj_mat=obj.faces_adj_mat)
    # print(target_variable)
    # print(node_features)


    # obj = PolyFeaturizer(vertices=verts_tetra,norm=True)
    # face_sides_features = encoder.face_sides_bin_encoder(obj.face_sides)
    # face_areas_features = obj.face_areas
    # node_features = np.concatenate([face_areas_features,face_sides_features],axis=1)

    # target_variable = obj.get_three_body_energy(nodes=obj.face_centers,
    #                                                 face_normals=obj.face_normals,
    #                                                 adj_mat=obj.faces_adj_mat)
    # print(target_variable)
    # print(node_features)


    # obj = PolyFeaturizer(vertices=verts_tetra_scaled,norm=True)
    # face_sides_features = encoder.face_sides_bin_encoder(obj.face_sides)
    # face_areas_features = obj.face_areas
    # node_features = np.concatenate([face_areas_features,face_sides_features],axis=1)

    # target_variable = obj.get_three_body_energy(nodes=obj.face_centers,
    #                                                 face_normals=obj.face_normals,
    #                                                 adj_mat=obj.faces_adj_mat)
    # print(target_variable)
    # print(node_features)















    # obj = PolyFeaturizer(vertices=verts_tetra)
    # print(obj.face_areas)

    # # Creating node features
    # face_sides_features = encoder.face_sides_bin_encoder(obj.face_sides)
    # face_areas_features = obj.face_areas
    # node_features = np.concatenate([face_areas_features,face_sides_features,],axis=1)

    # # Creating edge features
    # dihedral_angles = obj.get_dihedral_angles()

    # dihedral_angles_features = encoder.gaussian_continuous_bin_encoder(values = dihedral_angles, min_val=np.pi/4, max_val=np.pi, sigma= 0.2)
    
    # print(dihedral_angles_features)
    # # edge_features = dihedral_angles




    # print(len(obj.face_centers))
    # print(len(obj.vertices))
    # print(len(obj.face_sides))
    # print(pos)
    # obj.get_three_body_energy(pos,adj_mat)
    # pos = obj.face_normals
    # adj_mat = obj.faces_adj_mat

    # print(pos)
    # obj.get_three_body_energy(pos,adj_mat)
    # vert_adj_matrix = obj.verts_adj_mat
    # face_adj_matrix = obj.faces_adj_mat
    # print(obj.get_dihedral_angles())
    # print(vert_adj_matrix)
    # print(face_adj_matrix)
    # print(obj.edges)
    # print(obj.face_edges)
    # print(obj.face_edges)
    # x, edge_index, edge_attr, y, pos = obj.get_pyg_faces()
    # print(x.shape)
    # print(obj.get_dihedral_angles())
    # plotter = pv.Plotter()
    # plotter.add_mesh(obj.vertices,color="red",render_points_as_spheres=True)
    # plotter.add_mesh(obj.face_centers,color="green",render_points_as_spheres=True)
    # plotter.add_mesh(obj.edge_centers,color="blue",render_points_as_spheres=True)
    # # plot_adjacency(plotter,adj_matrix=vert_adj_matrix,points=obj.poly.vertices)
    # plot_adjacency(plotter,adj_matrix=face_adj_matrix,points=obj.face_centers)
    # plotter.show()