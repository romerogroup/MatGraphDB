import os

from poly_graphs_lib.data.create_datasets import create_material_polyhedra_dataset_2,create_material_polyhedra_dataset_3
from voronoi_statistics.voronoi_structure import VoronoiStructure
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def mp_cif_to_raw_poly:
    polyhedra_verts = []

    # Material polyhedra
    cif_files = os.listdir(mpcif_data_dir)
    for cif_file in cif_files:
        # print(cif_file)
        mp_id = cif_file.split('.')[0]

        try:
            voronoi_structure = VoronoiStructure(structure_id = f'{mpcif_data_dir}{os.sep}{cif_file}', 
                                            database_source='mp',
                                            database_id=mp_id,
                                            neighbor_tol=0.1)
        except Exception as e:
            print(e)
            continue
            # print(f'{cif_file} failed')
        voronoi_structure_dict = voronoi_structure.as_dict()

        for polyhedra_dict in voronoi_structure_dict['voronoi_polyhedra_info']:
            polyhedra_verts.append(np.array(polyhedra_dict['vertices']))

class External_to_RawConfig:

    data_dir = f"{PROJECT_DIR}{os.sep}datasets"
    raw_dir : str = f"{data_dir}{os.sep}raw"
    interim_dir : str = f"{data_dir}{os.sep}interim"
    external_dir : str = f"{data_dir}{os.sep}external"

    def __init__(self):
        if self.node_type not in ['face', 'vert']:
            raise ValueError("node_type must be either 'face' or 'vert'")

class DataGenerator:

    def __init__(self):
        self.config = DataGeneratorConfig() 

    def initialize_ingestion(self):
        feature_dir = data_dir + os.sep + y_val + os.sep + 'material_polyhedra' + os.sep + node_type + '_nodes'
        if os.path.exists(feature_dir):
            shutil.rmtree(feature_dir)
        os.makedirs(feature_dir)

        train_dir = f"{feature_dir}{os.sep}train"
        test_dir = f"{feature_dir}{os.sep}test"


        # Creating train,text, val directory
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir)

        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir)

        polyhedra_verts = []

        # Material polyhedra
        cif_files = os.listdir(mpcif_data_dir)
        for cif_file in cif_files:
            # print(cif_file)
            mp_id = cif_file.split('.')[0]

            try:
                voronoi_structure = VoronoiStructure(structure_id = f'{mpcif_data_dir}{os.sep}{cif_file}', 
                                                database_source='mp',
                                                database_id=mp_id,
                                                neighbor_tol=0.1)
            except Exception as e:
                print(e)
                continue
                # print(f'{cif_file} failed')
            voronoi_structure_dict = voronoi_structure.as_dict()

            for polyhedra_dict in voronoi_structure_dict['voronoi_polyhedra_info']:
                polyhedra_verts.append(np.array(polyhedra_dict['vertices']))


        # print(polyhedra_verts)

        verts_tetra_rot = verts_tetra.dot(rot_z(theta=25))*2 + 1

        test_poly = []
        test_poly.extend([verts_tetra, verts_cube,verts_oct,verts_dod, verts_tetra_rot])
        test_poly.extend([verts_mp567387_Ti,verts_mp4019_Ti,verts_mp3397_Ti,verts_mp15502_Ba,verts_mp15502_Ti])
        data_indices = np.arange(0,len(polyhedra_verts))
        poly_name = ['tetra','cube','oct','dod','rotated_tetra',
                    'dod-like','cube-like','tetra-like','cube-like','oct-like']
        
        test_indices = np.arange(0,len(test_poly))
        print("Processing Train polyhedra")
        process_polythedra(polyhedra_verts=polyhedra_verts,indices=data_indices, save_dir=train_dir,node_type=node_type,y_val=y_val)
        print("Processing Test polyhedra")
        process_polythedra(polyhedra_verts=test_poly,indices=test_indices, save_dir=test_dir ,node_type=node_type,y_val=y_val,labels=poly_name)


        create_material_polyhedra_dataset_3(data_dir=self.config.dataset_dir,
                                        mpcif_data_dir=self.config.mpcif_data_dir,
                                        node_type=self.config.node_type, 
                                        val_size=self.config.val_size,
                                        y_val = self.config.y_val)
        
if __name__ == "__main__":
    obj = DataGenerator()
    obj.initialize_ingestion()