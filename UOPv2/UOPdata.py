import random
#import numpy as np
#import glob
import os
#import sys
import re
from typing import List
#from scipy.spatial.distance import pdist,squareform
#from ase import Atoms
#from ase.geometry import geometry
#from ase.neighborlist import neighbor_list
#from .XYZdata import XYZ_reader

class Data_Feeder:
    def __init__(
            self,
            dir_prefixes: List[str],
            each_system_batch=10, max_neighbor=80, cutoff = 5, Traning = False):
        self.dir_prefixes = dir_prefixes
        self.files_in_each_dir = []
        #self.total_path = total_path
        self.each_system_batch = each_system_batch
        self.max_neighbor = max_neighbor
        self.cutoff = cutoff
        self.dictionary = {'MASK':0, 'C':1, 'O':2, 'N':3, 'H':4, 'CLAS':5, 'TEMP':6, 'PRESS':7}
        #self.crystal = {'liquid':0, 'P'}
        self.Traning = Traning
        self._collect_files()

    def _collect_files(self):
        self.files_in_each_dir = []
        for dir_prefix in self.dir_prefixes:
            file_list = []
            for root, dirs, files in os.walk(dir_prefix):
                for file in files:
                    if file.endswith('.xyz'):
                        filepath = os.path.join(root, file)
                        file_list.append(filepath)
            self.files_in_each_dir.append(file_list)
    
    def get_files(self) -> List[List[str]]:
        return self.files_in_each_dir
        
    def get_next_filename(filepath, step=1000):
        dirpath, filename = os.path.split(filepath)
        parts = filename.split('.')
        if len(parts) < 4:
            raise ValueError("Form of filename is unvail!")
        try:
            num = int(parts[-3])
        except ValueError:
            raise ValueError(f"ERROR in filename: {parts[-3]}")
        next_num = num + step
        parts[-3] = str(next_num)
        next_filename = '.'.join(parts)
        next_filepath = os.path.join(dirpath, next_filename)
        return next_filepath


    def read_lable(self,filename):
        filename = str(filename)
        label = int(filename.split('/')[-1].split('.')[0])
        temp  = float(filename.split('/')[-1].split('.')[1])
        press = float(filename.split('/')[-1].split('.')[2])
        time  = int(filename.split('/')[-1].split('.')[3])
        nextfile= get_next_filename(filename)
        return (label.astype(np.int32),
                temp.astype(np.float32),
                press.astype(np.float32),
                time.astype(np.int32),
                str(nextfile))

    def Generator_Dataset(self):
        sampled_files = []
        for file_list in 
#        while (len(idn)<self.each_system_batch):
#            path = 


# 示例
#filepath1 = '../Data/Paracetamol/Form_I/300.0.0/1.300.0.2000.Paracetamol.xyz'
#filepath2 = '../Data/Urea/Form_III/std/6.0.0.0.Urea.xyz'
#
#print(get_next_filename(filepath1))  # ../Data/Paracetamol/Form_I/300.0.0/1.300.0.3000.Paracetamol.xyz
#print(get_next_filename(filepath2))  # ../Data/Urea/Form_III/std/6.0.0.1000.Urea.xyz

#filename1 = r"../Data/Paracetamol/Form_I/300.0.0/1.300.0.2000.Paracetamol.xyz"
#filename1 = r" ../Data/Urea/Form_III/std/6.0.0.0.Urea.xyz"
#print(filename1)
#print(get_next_filename(filename1))

if __name__ == "__main__":
    dir_prefixes = ["../Data/Paracetamol/", "../Data/Urea/"]
    colletor = Data_Feeder(dir_prefixes) 
    #colletor.collect_files()
    files_in_each_dir = colletor.get_files()

    for idx, file_list in enumerate(files_in_each_dir):
        print(f"{idx+1} :（{dir_prefixes[idx]}）files：")
        for f in file_list:
            print(f"  {f}")
