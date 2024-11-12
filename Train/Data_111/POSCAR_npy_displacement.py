import os
import glob
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure
import numpy as np

POSCAR_path = glob.glob('./displacement/*.POSCAR')
element_map = {'C':0, 'O':1}

def write_POSCAR_npy(path,save_name):
    path = str(path)
    save_name = str(save_name)
    save_name = save_name.split('.POSCAR')[0]
    data = []
    #save_path_dist = './dist/' + save_name + '.npy'
    save_path_coord = './coord/' + save_name + '.npy'
    print(path)
    structure = Structure.from_file(path)
    #np.save(save_path_dist, structure.distance_matrix)
    for site in structure.sites:
        row = [element_map[str(site.specie)],site.coords[0],site.coords[1],site.coords[2]]
        data.append(row)
    np.save(save_path_coord,data)

#write_POSCAR_npy("./POSCAR/1.POSCAR",'1')

for i in range(len(POSCAR_path)):
    #label_type= (POSCAR_path[i]).split('R/')[-1].split('.')[0]
    #cryst_name = (POSCAR_path[i]).split('R/')[-1].split('.')[-2]
    file_name = (POSCAR_path[i]).split('displacement/')[-1]
    write_POSCAR_npy(POSCAR_path[i],file_name)


