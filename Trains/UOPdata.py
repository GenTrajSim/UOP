import random
import numpy as np
import glob
import os
import sys
import re
#from scipy.spatial.distance import pdist,squareform
from ase import Atoms
from ase.geometry import geometry
from ase.neighborlist import neighbor_list

test_path = "./Data/test/model.xyz"

#data_test = np.loadtxt(test_path)
#print(data_test)

class XYZ_reader:
    def __init__(self,filename):
        self.filename = filename
        self.Natom = None
        self.cell = None
        self.ase_atoms = None
        self.coords = None
        self.elements = None
        self.elements_ids = None
        self.dictionary = {'MASK':0, 'C':1, 'O':2, 'N':3, 'H':4, 'CLAS':5, 'TEMP':6, 'PRESS':7}
        self._read_file()
    def _read_file(self):
        with open(self.filename,'r') as f:
            lines = f.readlines()
        self.Natom = int(lines[0].strip())
        lattice_line = lines[1]
        match = re.search(r'Lattice="([^"]+)"', lattice_line)
        if match:
            cell_str = match.group(1)
            self.cell = np.array([float(x) for x in cell_str.split()])
            self.cell = self.cell.reshape(3,3)
        else:
            raise ValueError("Cannot find Lattice information.")
        #print(self.cell)
        atom_lines = lines[2:]
        self.ase_atoms = []
        self.elements = []
        self.coords = []
        for line in atom_lines:
            parts = line.split()
            if not parts or parts[0] == 'H':
                continue
            self.elements.append(parts[0])
            self.coords.append([float(x) for x in parts[1:4]])
        self.ase_atoms = Atoms(
                    symbols=self.elements,
                    positions=self.coords,
                    cell=self.cell,
                    pbc=True
                )
        self.coords = np.array(self.coords)
        self.elements = np.array(self.elements)
        self.elements_ids = np.array([self.dictionary[e1] for e1 in self.elements], dtype=int)
    def get_cell(self):
        return self.cell

    def get_num_atoms(self):
        return self.Natom

    def get_ase_atom(self):
        return self.ase_atoms

    def get_coords(self):
        return self.coords

    def get_elements(self):
        return self.elements

    def get_elements_ids(self):
        return self.elements_ids

    def get_local_env(self, max_neighbor=50, cutoff=10.0):
        local_token = []
        local_coord = []


parser = XYZ_reader(test_path)
print("Natoms:", parser.get_num_atoms())
print("cell:", parser.get_cell().shape)
#print("ATOMS:\n", parser.get_ase_atom())
print("ele:\n",parser.get_elements_ids().shape)
print("coord:\n", parser.get_coords().shape)

idx_i,idx_j,offsets = neighbor_list('ijS',parser.get_ase_atom(),10)
print("idx_i:\n",idx_i)
print("idx_j:\n",idx_j)
print("offsets:\n",offsets)
