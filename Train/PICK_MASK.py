import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import numpy as np
import glob
import os
import sys
AUTO = tf.data.experimental.AUTOTUNE
from scipy.spatial.distance import pdist,squareform

class Data_Pre:
  def __init__(self, total_std_path, total_var_path, picknoise = 2, maxatom=50,  d_cut = 5,  len_dictionary = 5,  Tranining = False):
    self.total_path = total_std_path
    self.total_var_path = total_var_path
    self.picknoise = picknoise
    self.maxatom = maxatom
    self.d_cut = d_cut
    self.len_dictionary = len_dictionary
    self.Tranining = Tranining
    
  def indices_distance (self, dist, N_atom, d):
    indices = []
    for i in range(N_atom):
      row = dist[i,:]
      indices.append(np.where(row<d)[0])
    return indices
  
  def indices_distance_d (self, dist, N_atom, d, max_small_index):
    indices = []
    for i in range(N_atom):
      row = dist[i,:]
      small_indices = np.where(row < d)[0]
      sorted_indices = np.argsort(row[small_indices])[:max_small_index]
      indices.append(small_indices[sorted_indices])
    return indices
    
  def pick_not_replet (self, path, ni):
    selected_paths = random.sample(list(path), ni)
    return selected_paths
  
  def read_npy_file(self, filename1, filename2, filename_list, ni):
    filename1 = str(filename1)
    filename2 = str(filename2)
    select_ni = None
    if self.Tranining == True:
      select_ni = pick_not_replet(filename_list,ni)
      tf.print(filename_list[i])
    tf.print(filename1)
    tf.print(filename2)
    label = int(filename.split('d/')[1].split('.')[0])  #label
    data = np.load(filename1)
    coord = data[:,1:4]                                 #coord
    token = data[:,0].astype(int)+1                     #token
    if self.Training == False :
      data2 = np.load(filename2)                        #dist
    else:
      data2 = squareform(pdist(coord,'euclidean'))      #dist
    atom_N = data.shape[0]
    indices = self.indices_distance_d(data2,atom_N,8,50) ####
