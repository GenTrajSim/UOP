import numpy as np
import re
from ase import Atoms
from ase.neighborlist import neighbor_list

#test_path = "./Data/test/model.xyz"

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
        self.Natom = self.elements.shape[0]
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

    def get_local_env(self, max_neighbor=80, cutoff=5):
        local_token = []
        local_coord = []
        local_token = np.full((self.Natom,max_neighbor+4,1),self.dictionary['MASK'],dtype=int)
        local_token[:,0,0] = 5
        local_token[:,1,0] = 6
        local_token[:,2,0] = 7
        local_coord = np.zeros((self.Natom,max_neighbor+4,3),dtype=float)
        #local_coord[:,0:4,:] = 0
        idx_i,idx_j,offsets = neighbor_list('ijS',self.ase_atoms,cutoff)
        neighbor_dict = {i: [] for i in range(self.Natom)}
        for i,j,S in zip(idx_i,idx_j,offsets):
            neighbor_pos = self.coords[j] + np.dot(S,self.cell)
            rel_pos = neighbor_pos - self.coords[i]
            neighbor_dict[i].append((j,rel_pos))
        for i in range(self.Natom):
            neighors = sorted(neighbor_dict[i], key=lambda x: np.linalg.norm(x[1]))
            local_token[i,3,0] = self.elements_ids[i]
            for k, (j, rel_pos) in enumerate(neighors[:max_neighbor]):
                local_token[i,k+4,0] = self.elements_ids[j]
                local_coord[i,k+4,:]=rel_pos
        return (local_token.astype(np.int32), 
                local_coord.astype(np.float32))

if __name__ == "__main__":
    test_path = "./Data/test/model.xyz"
    parser = XYZ_reader(test_path)
    print("Natoms:", parser.get_num_atoms())
    print("cell:", parser.get_cell().shape)
    #print("ATOMS:\n", parser.get_ase_atom())
    print("ele:\n",parser.get_elements_ids()[0])
    print("coord:\n", parser.get_coords().shape)
    local_elem, local_coords = parser.get_local_env(80,10)
    local_elem = np.array(local_elem)
    local_coords=np.array(local_coords)
    print(local_elem.shape)
    print(local_coords.shape)
#idx_i,idx_j,offsets = neighbor_list('ijS',parser.get_ase_atom(),10)
#print("idx_i:\n",idx_i.shape)
#print("idx_j:\n",idx_j.shape)
#print("offsets:\n",offsets.shape)
#neighbor_dict = {i: [] for i in range(parser.get_num_atoms())}
#for i,j,S in zip(idx_i,idx_j,offsets):
#    test_neighbor_pos = parser.get_coords()[j] + np.dot(S,parser.get_cell())
#    rel_pos = test_neighbor_pos - parser.get_coords()[i]
#    neighbor_dict[i].append((j,rel_pos))
#    print(rel_pos)
#neighbor_dict = np.array(neighbor_dict)
#def get_shape(lst):
#    shape = []
#    while isinstance(lst, list):
#        shape.append(len(lst))
#        if len(lst) == 0:
#            break
#        lst = lst[0]
#    return tuple(shape)
#
#lst3 = [[[1], [2]], [[3], [4]]]
#print(get_shape(lst3))  # 输出 (2, 2, 1)

#print(get_shape(neighbor_dict[0]))

#class Data_Feeder:
#    def __init__(self, total_path, each_system_batch=2, max_neighbor=80, cutoff = 5, Traning = False):
#        self.total_path = total_path
#        self.each_system_batch = each_system_batch
#        self.max_neighbor = max_neighbor
#        self.cutoff = cutoff
#        self.dictionary = {'MASK':0, 'C':1, 'O':2, 'N':3, 'H':4, 'CLAS':5, 'TEMP':6, 'PRESS':7}
#        #self.crystal = {'liquid':0, 'P'}
#        self.Traning = Traning
#    
#    def read_lable(self,filename):
#        filename = str(filename)
#        label = int(filename.split('/')[-1].split('.')[0])
#        temp  = float(filename.split('/')[-1].split('.')[1])
#        press = float(filename.split('/')[-1].split('.')[2])
#        time  = int(filename.split('/')[-1].split('.')[3])
#
