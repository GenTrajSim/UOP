import os
import glob
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure
import numpy as np
import sys

os.environ["MKL_NUM_THREADS"] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'

cal_path = 'only1'
cal_path = sys.argv[1]
cal_path_poscar = '../SAVE/MultiPT/'+cal_path+'/56.panding.POSCAR'
#POSCAR_path = glob.glob('../SAVE/MultiPT/P5000_T200/56.panding.POSCAR')
POSCAR_path = glob.glob(cal_path_poscar)
element_map = {'C':0, 'O':1}

def write_POSCAR_npy(path,save_name):
    path = str(path)
    save_name = str(save_name)
    save_name = save_name.split('.POSCAR')[0]
    data = []
    save_path_dist = '../SAVE/MultiPT/'+cal_path+'/dist/' + save_name + '.npy'
    save_path_coord = '../SAVE/MultiPT/'+cal_path+'/coord/' + save_name + '.npy'
    print(path)
    print(save_path_coord)
    structure = Structure.from_file(path)
    np.save(save_path_dist, structure.distance_matrix)
    for site in structure.sites:
        row = [element_map[str(site.specie)],site.coords[0],site.coords[1],site.coords[2]]
        data.append(row)
    np.save(save_path_coord,data)

#write_POSCAR_npy("./POSCAR/1.POSCAR",'1')

for i in range(len(POSCAR_path)):
    #label_type= (POSCAR_path[i]).split('R/')[-1].split('.')[0]
    #cryst_name = (POSCAR_path[i]).split('R/')[-1].split('.')[-2]
    file_name_p = '../SAVE/MultiPT/'+cal_path+'/'
    #file_name = (POSCAR_path[i]).split('../SAVE/MultiPT/P5000_T200/')[-1]
    file_name = (POSCAR_path[i]).split(file_name_p)[-1]
    write_POSCAR_npy(POSCAR_path[i],file_name)


