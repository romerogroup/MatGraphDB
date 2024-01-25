import json
import copy
import numpy as np
import itertools

from coxeter.shapes import ConvexPolyhedron
from matgraphdb.utils.shapes import PLUTONIC_POLYS
from matgraphdb.utils.math import similarity_score

from ..utils.math import gaussian_continuous_bin_encoder,softmax
from matgraphdb.cfg.coordination_geometries_files import cg_list, mp_symbols,mp_coord_encoding

class PolyFeaturizer:
    
    def __init__(self, vertices, norm:bool=True, coord_env=None):

        self.poly = ConvexPolyhedron(vertices=vertices)
        self.poly.merge_faces(atol=1e-4, rtol=1e-5)

        self.coord_env=coord_env
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

        self.get_shape_measures()
        self.get_coord_env_encoding()
        self.get_coord_encoding()


        return None
    
    def verts_edge_index(self):
        return self.edges
    
    def faces_edge_index(self):
        return self.face_edges
    
    def verts_energy_per_node(self):
        return self.get_energy_per_node(self.vertices,self.verts_adj_mat)
    
    def faces_energy_per_node(self):
        return self.get_energy_per_node(self.face_centers,self.faces_adj_mat)

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
        return np.array(edges)
    
    def get_vertices_adjacency(self):
        n = len(self.vertices)
        adj_matrix = np.zeros((n, n), dtype=int)
        for edge in self.edges:
            adj_matrix[edge[0], edge[1]] = 1
            adj_matrix[edge[1], edge[0]] = 1
        return adj_matrix
    
    def get_verts_neighbor_angles(self):
        n_verts = len(self.vertices)
        neighbor_angles = []
        neighbor_angle_weights = []
        for i_vert in range(n_verts):    
            neighbors = self.verts_adj_mat[i_vert,:]
            non_zero_indexes = np.nonzero(neighbors)[0]

            # For a given vertex find the combinations of 2 neighsbors to from the angle
            neighbor_permutations = list(itertools.combinations(non_zero_indexes,2))

            weight_angles = []
            tmp_angles=[]
            for neighbor_permutation in neighbor_permutations:
                i = neighbor_permutation[0]
                j = neighbor_permutation[1]

                #Finding the corresponding face area for the 3 points
                for i_face,face in enumerate(self.faces):
                    if i_vert in face and i in face and j in face:
                        face_area = self.face_areas[i_face]

                # Create vectors pointing to neighbors
                vec_a = self.vertices[i] - self.vertices[i_vert]
                vec_b = self.vertices[j] - self.vertices[i_vert]
                vec_a_norm = np.linalg.norm(vec_a)
                vec_b_norm = np.linalg.norm(vec_b)


                weight_angles.append(vec_a_norm * vec_b_norm * face_area)

                angle = np.arccos( np.around(np.dot(vec_a , vec_b)/(vec_a_norm*vec_b_norm), decimals=8) )
                tmp_angles.append(angle)

            norm = sum(weight_angles)
            weight_angles = np.array(weight_angles) / norm

            neighbor_angle_weights.append(weight_angles)
            neighbor_angles.append(tmp_angles)
        return neighbor_angles, neighbor_angle_weights

    def verts_neighbor_angles_encodings(self,n_bins=50, min_val=0, max_val=(3/2)*np.pi, sigma=0.2):
        neighbor_angles_list, neighbor_angle_weights = self.get_verts_neighbor_angles()
        n_verts = len(self.vertices)
        neighbor_angle_encodings = []
        for i in range(n_verts):
            neighbor_angles = neighbor_angles_list[i]
            neighbor_angles_weights = neighbor_angle_weights[i]
            neighbor_encoding = None
            for neighbor_angle,neighbor_angles_weight in zip(neighbor_angles,neighbor_angles_weights):
                # tmp_encoding =  gaussian_continuous_bin_encoder(values=neighbor_angle, min_val=min_val, max_val=max_val, sigma=sigma)
                tmp_encoding = gaussian_continuous_bin_encoder(values=neighbor_angles_weight * neighbor_angle,
                                                               n_bins=n_bins, 
                                                               min_val=min_val, 
                                                               max_val=max_val, 
                                                               sigma=sigma)
                if neighbor_encoding is None:
                    neighbor_encoding = tmp_encoding
                else:
                    neighbor_encoding += tmp_encoding

            neighbor_angle_encodings.append(neighbor_encoding)
        return np.array(neighbor_angle_encodings)
    
    def get_verts_neighbor_face_areas(self):
        n_verts = len(self.vertices)
        neighbor_areas = []
        for i_vert in range(n_verts):    
            neighbors = self.verts_adj_mat[i_vert,:]
            non_zero_indexes = np.nonzero(neighbors)[0]

            # For a given vertex find the combinations of 2 neighsbors to from the angle
            neighbor_permutations = list(itertools.combinations(non_zero_indexes,2))

            weight_angles = []
            tmp_areas=[]
            for neighbor_permutation in neighbor_permutations:
                i = neighbor_permutation[0]
                j = neighbor_permutation[1]

                #Finding the corresponding face area for the 3 points
                for i_face,face in enumerate(self.faces):
                    if i_vert in face and i in face and j in face:
                        face_area = self.face_areas[i_face]
                tmp_areas.append(face_area)
            neighbor_areas.append(tmp_areas)

        return neighbor_areas
    
    def verts_neighbor_areas_encodings(self,n_bins=100,min_val=0, max_val=3, sigma=0.1):
        neighbor_areas_list = self.get_verts_neighbor_face_areas()
        n_verts = len(self.vertices)
        neighbor_areas_encodings = []
        for i in range(n_verts):
            neighbor_angles = neighbor_areas_list[i]
            neighbor_encoding = None
            for neighbor_angle in neighbor_angles:
                tmp_encoding = gaussian_continuous_bin_encoder(values= neighbor_angle, 
                                                               n_bins=n_bins,
                                                               min_val=min_val, 
                                                               max_val=max_val, 
                                                               sigma=sigma)
                if neighbor_encoding is None:
                    neighbor_encoding = tmp_encoding
                else:
                    neighbor_encoding += tmp_encoding
            # neighbor_encoding=neighbor_encoding/max(neighbor_encoding)
            neighbor_areas_encodings.append(neighbor_encoding)

        return np.array(neighbor_areas_encodings)

    def verts_neighbor_distance(self,do_encoding=False, n_bins=100,min_val=0, max_val=3, sigma=0.1 ):
        
        if do_encoding:
            neighbor_distances = np.zeros(shape =(len(self.edges),n_bins))
        else:
            neighbor_distances = np.zeros(shape =(len(self.edges),1))
        
        for index,edge in enumerate(self.edges):
            i = edge[0]
            j = edge[1]

            distance = np.linalg.norm(self.vertices[i] - self.vertices[j])
            if do_encoding:
                distance=gaussian_continuous_bin_encoder(values=distance,
                                                n_bins=n_bins, 
                                                min_val=min_val, 
                                                max_val=max_val, 
                                                sigma=sigma)
            
            neighbor_distances[index] = distance
        return neighbor_distances

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
    
    def get_coord_env_encoding(self):
        tmp_dict=copy.copy(mp_symbols)
        if self.coord_env:
            symbol=self.coord_env['ce_symbol']
            tmp_dict[symbol]+=1
            self.coord_env_encoding=[ value for key,value in tmp_dict.items()]

        return self.coord_env_encoding
    
    def get_coord_encoding(self):
        if self.coord_env:
            symbol=self.coord_env['ce_symbol']
            self.coord_encoding=mp_coord_encoding[symbol]
        return self.coord_encoding
    
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
    
    def get_shape_measures(self,loss=None,max_iter=100,alpha=1,threshold_plan =0):
        
        shape_measures=np.zeros(shape=len(PLUTONIC_POLYS))
        for i,plutonic_poly in enumerate(PLUTONIC_POLYS):
            score,std = similarity_score(point_set_1=self.vertices, 
                            point_set_2=plutonic_poly[0], 
                            loss=loss,
                            max_iter=max_iter,
                            alpha=alpha,
                            threshold_plan=threshold_plan)
            shape_measures[i]=score

        self.shape_measures_inv=1/shape_measures
        # self.shape_measures_inv[np.isnan(self.shape_measures_inv)] = 1
        self.shape_measures_inv[np.isinf(self.shape_measures_inv)] = 10000
        self.shape_measures_norm =self.shape_measures_inv/self.shape_measures_inv.sum()
        self.shape_measures_softmax=softmax(self.shape_measures_inv)
        self.shape_measures=shape_measures
        return self.shape_measures,self.shape_measures_norm,self.shape_measures_softmax
    
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