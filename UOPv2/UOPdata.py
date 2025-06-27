import random
import numpy as np
#import glob
import os
#import sys
import re
from typing import List
#from scipy.spatial.distance import pdist,squareform
#from ase import Atoms
#from ase.geometry import geometry
#from ase.neighborlist import neighbor_list
from .XYZdata import XYZ_reader

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
        
    def get_next_filename(self, filepath, step=1000):
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
        nextfile= self.get_next_filename(filename)
        return (np.int32(label),
                np.float32(temp),
                np.float32(press),
                np.int32(time),
                str(nextfile))

    def Generator_Dataset(self):
        sampled_files = []
        local_elements_c = None
        local_coords_c = None
        local_elements_n = None
        local_coords_n = None
        local_label = None
        local_temp = None
        local_press = None
        local_pred_t = None
        for file_list in self.files_in_each_dir:
            if len(file_list) <= self.each_system_batch:
                sampled = file_list
            else:
                sampled = random.sample(file_list,self.each_system_batch)
            sampled_files.extend(sampled)
        for f in sampled_files:
            (label,temp,press,time,nextf) = self.read_lable(f)
            print(label,temp,press,time,nextf)
            parser_current = XYZ_reader(f)
            (local_elements_c, local_coords_c) = parser_current.get_local_env(self.max_neighbor,self.cutoff)
            print(local_elements_c.shape)
            print(local_coords_c.shape)
            if os.path.exists(nextf):
                print("next file exist.")
                parser_next = XYZ_reader(nextf)
                (local_elements_n, local_coords_n) = parser_next.get_local_env(self.max_neighbor,self.cutoff)
                predict_t = 1
                #print(local_elements_n.shape)
                #print(local_coords_n.shape)
            else:
                print("without next file!")
                (local_elements_n, local_coords_n) = (local_elements_n, local_coords_n)
                predict_t = 0
                #print(local_elements_n[4])
                #print(local_coords_n[4])
            local_label = np.full((local_elements_c.shape[0],),label)
            local_temp =  np.full((local_elements_c.shape[0],),temp)
            local_press = np.full((local_elements_c.shape[0],),press)
            local_pred_t= np.full((local_elements_c.shape[0],),predict_t)
            print(local_label.shape)
            print(local_temp.shape)
            print(local_press.shape)
            print(local_pred_t)
            for atom in range(local_label.shape[0]):
                yield local_label[atom],local_temp[atom],local_press[atom],local_pred_t[atom],local_elements_c[atom],local_elements_n[atom],local_coords_c[atom],local_coords_n[atom]
        #return sampled_files

    def Generate_batch(self, batch_size, Repeat_size, shuffle_size):
        dataset = tf.data.Dataset.from_generator(
                self.Generator_Dataset,
                output_types=(tf.int32,tf.float32,tf.float32, tf.int32, tf.int32,tf.int32, tf.float32, tf.float32), 
                output_shapes=((),(),(),(),(self.max_neighbor+4,1),(self.max_neighbor+4,1),(self.max_neighbor+4,3),(self.max_neighbor+4,3))
                )
        dataset = dataset.repeat(Repeat_size)
        dataset = dataset.shuffle(shuffle_size).batch(batch_size)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset
    
    def Generate_test_batch(self, batch_size):
        dataset = tf.data.Dataset.from_generator(
                self.Generator_Dataset,
                output_types=(tf.int32,tf.float32,tf.float32, tf.int32, tf.int32,tf.int32, tf.float32, tf.float32),
                output_shapes=((),(),(),(),(self.max_neighbor+4,1),(self.max_neighbor+4,1),(self.max_neighbor+4,3),(self.max_neighbor+4,3))
                )
        #dataset = dataset.repeat(Repeat_size)
        dataset = dataset.batch(batch_size)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

        #return sampled_files
#        while (len(idn)<self.each_system_batch):
#            path = 

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
    dir_prefixes = ["./Data/Paracetamol/", "./Data/Urea/"]
    colletor = Data_Feeder(dir_prefixes,cutoff = 8,each_system_batch=2) 
    #colletor.collect_files()
    files_in_each_dir = colletor.get_files()

    for idx, file_list in enumerate(files_in_each_dir):
        print(f"{idx+1} :（{dir_prefixes[idx]}）files：")
        for f in file_list:
            print(f"  {f}")

    print("===================================")
    #collletor.Generator_Dataset()
    batch_data_test = colletor.Generate_test_batch(10)
    for (batch, (local_label,local_temp,local_press,local_pred_t,local_elements_c,local_elements_n,local_coords_c,local_coords_n)) in enumerate(batch_data_test):
        tf.print(
                local_label.shape,
                local_temp.shape,
                local_press.shape,
                local_pred_t.shape,
                local_elements_c.shape,
                local_elements_n.shape,
                local_coords_c.shape,
                local_coords_n.shape)
    #for f in sampled_files:
    #    print(f)
        #print(get_next_filename(f))
    #np_sample = np.array(sampled_files)
    #for f in np_sample:
    #    print(f)

