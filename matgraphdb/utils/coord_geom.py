import os
import numpy as np
import glob
import json

from matgraphdb.utils import PKG_DIR


coord_geom_dir=os.path.join(PKG_DIR,'utils','coordination_geometries')

files=glob.glob(coord_geom_dir + '/*.json')

cg_list=[]
mp_symbols={}
cg_points={}
mp_coord_encoding={}

for file in files:

    with open(file) as f:
        dd = json.load(f)
    cg_list.append(dd)
    mp_symbols.update({dd['mp_symbol']:0})
    cg_points.update({dd['mp_symbol']:dd['points']})


    coord_encoding=np.zeros(shape=14)
    coord_num=int(dd['mp_symbol'].split(':')[-1])
    
    if coord_num<=13:
        coord_encoding[coord_num-1]=1
    elif coord_num==20:
        coord_encoding[-1]=1
    mp_coord_encoding.update({dd['mp_symbol']:coord_encoding})

coord_nums=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,20])

