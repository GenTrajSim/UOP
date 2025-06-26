#import random
#import numpy as np
#import glob
#import os
#import sys
#import re
#from scipy.spatial.distance import pdist,squareform
#from ase import Atoms
#from ase.geometry import geometry
#from ase.neighborlist import neighbor_list
from .XYZdata import XYZ_reader

class Data_Feeder:
    def __init__(self, total_path, each_system_batch=2, max_neighbor=80, cutoff = 5, Traning = False):
        self.total_path = total_path
        self.each_system_batch = each_system_batch
        self.max_neighbor = max_neighbor
        self.cutoff = cutoff
        self.dictionary = {'MASK':0, 'C':1, 'O':2, 'N':3, 'H':4, 'CLAS':5, 'TEMP':6, 'PRESS':7}
        #self.crystal = {'liquid':0, 'P'}
        self.Traning = Traning

    def read_lable(self,filename):
        filename = str(filename)
        label = int(filename.split('/')[-1].split('.')[0])
        temp  = float(filename.split('/')[-1].split('.')[1])
        press = float(filename.split('/')[-1].split('.')[2])
        time  = int(filename.split('/')[-1].split('.')[3])

