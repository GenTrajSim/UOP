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

class Data_Feeder(XYZ_reader):
    def __init__(
            self,
            dir_prefixes: List[str],
            each_system_batch=10, max_neighbor=80, cutoff = 5, Traning = False):
        super(Data_Feeder).__init__()
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
            print("DATA_info:",label,temp,press,time,nextf)
            parser_current = XYZ_reader(f)
            (local_elements_c, local_coords_c) = parser_current.get_local_env(self.max_neighbor,self.cutoff)
            print("DATA_info:",local_elements_c.shape)
            print("DATA_info:",local_coords_c.shape)
            #if os.path.exists(nextf):
                #print("next file exist.")
                #parser_next = XYZ_reader(nextf)
                #(local_elements_n, local_coords_n) = parser_next.get_local_env(self.max_neighbor,self.cutoff)
                #predict_t = 1
                #print(local_elements_n.shape)
                #print(local_coords_n.shape)
            #else:
                #print("without next file!")
                #(local_elements_n, local_coords_n) = (local_elements_n, local_coords_n)
                #predict_t = 0
                #print(local_elements_n[4])
                #print(local_coords_n[4])
            local_label = np.full((local_elements_c.shape[0],),label)
            local_temp =  np.full((local_elements_c.shape[0],),temp)
            local_press = np.full((local_elements_c.shape[0],),press)
            #local_pred_t= np.full((local_elements_c.shape[0],),predict_t)
            print("DATA_info:",local_label.shape)
            print("DATA_info:",local_temp.shape)
            print("DATA_info:",local_press.shape)
            #print(local_pred_t)
            for atom in range(local_label.shape[0]):
                yield local_label[atom],local_temp[atom],local_press[atom],local_elements_c[atom],local_coords_c[atom]
        #return sampled_files

    def Generate_batch(self, batch_size, Repeat_size, shuffle_size):
        dataset = tf.data.Dataset.from_generator(
                self.Generator_Dataset,
                output_types=(tf.int32,tf.float32,tf.float32, tf.int32,tf.float32), 
                output_shapes=((),(),(),(self.max_neighbor+4,1),(self.max_neighbor+4,3))
                )
        dataset = dataset.repeat(Repeat_size)
        dataset = dataset.shuffle(shuffle_size).batch(batch_size)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset
    
    def Generate_test_batch(self, batch_size):
        dataset = tf.data.Dataset.from_generator(
                self.Generator_Dataset,
                output_types=(tf.int32,tf.float32,tf.float32,tf.int32, tf.float32),
                output_shapes=((),(),(),(self.max_neighbor+4,1),(self.max_neighbor+4,3))
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

class create_masks:
    def __init__(self, noise_C,token_noise=0.2, iterT = 50, training=True):
        self.noise_C = noise_C
        self.token_noise = token_noise
        #self.noise_T = noise_T
        self.iterT = iterT
        self.training = training
        self.dictionary = {'MASK':0, 'C':1, 'O':2, 'N':3, 'H':4, 'CLAS':5, 'TEMP':6, 'PRESS':7}
    def Create_noise(self, tokens_c, coords_c):
        bsz = tf.shape(tokens_c)[0]
        Natom=tf.shape(tokens_c)[1]
        tokens_c = tf.reshape(tokens_c, [bsz,Natom])
        NO_padding_mask = tf.cast(tf.not_equal(tokens_c, 0), dtype=tf.int32)
        NO_padding_clas = tf.cast(tf.not_equal(tokens_c, 5), dtype=tf.int32)
        NO_padding_temp = tf.cast(tf.not_equal(tokens_c, 6), dtype=tf.int32)
        NO_padding_pres = tf.cast(tf.not_equal(tokens_c, 7), dtype=tf.int32)
        NO_padding = NO_padding_mask*NO_padding_clas*NO_padding_temp*NO_padding_pres
        #
        random_MASK_Prob = tf.random.uniform(shape=NO_padding_mask.shape) < self.token_noise
        weight_token = tf.cast((tf.random.uniform(shape=NO_padding.shape))*(len(self.dictionary)-4)+1, dtype=tf.int32)
        update_token = weight_token*tf.cast(random_MASK_Prob,dtype=tf.int32)*NO_padding
        update_token = tf.where(tf.not_equal(update_token,0), update_token, tokens_c)
        #
        NO_padding = tf.tile(tf.expand_dims(NO_padding,axis=-1),[1,1,3])
        #
        #tf.print("NO_padding",NO_padding)
        ###########################
        #Pred_t = tf.ones(shape=(bsz,1),dtype=tf.int32)
        if self.training:
            iter_i = tf.random.uniform(shape=(bsz,1), minval=1, maxval=self.iterT+1, dtype=tf.int32)
            #Pred_t = tf.random.uniform(shape=(bsz,1), minval=0, maxval=2, dtype=tf.int32)
        else:
            iter_i = tf.cast( tf.fill([bsz, 1], self.iterT), dtype=tf.int32)
        noise = tf.random.normal(shape=(bsz,Natom,3))
        noise = noise*tf.cast(NO_padding,dtype=tf.float32)
        ## ca dist
        #dist = None
        #dist = tf.norm(coords_c, axis=2, keepdims=True)
        ## No eq noise
        #noise = noise #* dist * 0.2
        #tf.print("dist:=====")
        #tf.print(tf.reduce_mean(dist))
        #
        #tf.print("orgion:",pred_t_i)
        #Pred_t = tf.reshape(Pred_t,(bsz,))
        #Pred_t = Pred_t*pred_t_i
        ##
        betas = np.linspace(1e-5, self.noise_C, self.iterT+1,dtype=np.float32) #self.noise_C=0.02
        alphas= 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        alpha_bars_tf = tf.convert_to_tensor(alpha_bars, dtype=tf.float32)
        t_indices = tf.squeeze(iter_i, axis=1)
        alpha_bar_t = tf.gather(alpha_bars_tf, t_indices)  # shape=(batch_size,)
        alpha_bar_t = tf.reshape(alpha_bar_t, (bsz, 1, 1))
        input_coords = (
                tf.sqrt(alpha_bar_t)*coords_c +
                tf.sqrt(1.0-alpha_bar_t) * noise
                )
        ##
        #tf.print("orgion:",Pred_t)
        #tf.print(Pred_t.shape)
        #Pick_out_coord =tf.tile(Pred_t[:, None, None], [1, Natom, 3])  
        #Pred_t = tf.reshape(Pred_t,(bsz,))
        iter_i = tf.reshape(iter_i,(bsz,))
        #tf.print("change:",Pick_out_coord)
        #output_coords = tf.where(Pick_out_coord==1,coords_n,coords_c)
        return (
                update_token,input_coords,noise,iter_i
                )

## need test ##############################################################################################################

class DiffusionSampler:
    def __init__(self, score_model, structure_shape, noise_C=0.02, iterT=50):
        self.score_model = score_model
        self.structure_shape = structure_shape
        self.iterT = iterT
        self.betas = tf.linspace(1e-5, noise_C, iterT)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = tf.math.cumprod(self.alphas, axis=0)
    def sample(self, token, x_t=None):
        if x_t is None:
            x_t = tf.random.normal(self.structure_shape)
        for t in reversed(range(1, self.iterT + 1)):
            beta_t = self.betas[t-1]
            alpha_t = self.alphas[t-1]
            alpha_bar_t = self.alpha_bars[t-1]
            bsz = token.shape[0]
            t_tensor = tf.cast( tf.fill([bsz, 1], t), dtype=tf.int32)
            t_tensor = tf.reshape(t_tensor,(bsz,))
            #t_tensor = tf.fill([self.structure_shape[0]], t) if len(self.structure_shape) > 1 else tf.constant([t])
            out = self.score_model(token, x_t, t_tensor, training=False)
            (pred_tokens, _, _, _, pred_noise, _, _) = out
            pred_tokens = tf.argmax(pred_tokens, axis=-1) 
            token = pred_tokens
            coef1 = 1.0 / tf.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / tf.sqrt(1.0 - alpha_bar_t)
            x_prev = coef1 * (x_t - coef2 * pred_noise)
            if t > 1:
                sigma_t = tf.sqrt(beta_t)
                x_prev += sigma_t * tf.random.normal(tf.shape(x_t))
            # Debug打印
            tf.print("t:", t, "x_t mean/std:", tf.reduce_mean(x_t), tf.math.reduce_std(x_t),
                     "pred_noise mean/std:", tf.reduce_mean(pred_noise), tf.math.reduce_std(pred_noise),
                     "beta_t:", beta_t, "alpha_t:", alpha_t, "sqrt_alpha_t:", tf.sqrt(alpha_t))
            if t == 99:
                tf.print(x_prev,summarize=500000)
            x_t = x_prev
        return x_t
            #
## need test ##############################################################################################################

# Samples
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
    filename_c_old = 'oldcoords.xyz'
    filename_c_new = 'newcoords.xyz'
    filename_e_old = 'oldelement.xyz'
    filename_e_new = 'newelement.xyz'
    filename_log = 'traj.log'
    filename_c_noise = 'noise.xyz'
    ##
    fco = open(filename_c_old,'a')
    fcn = open(filename_c_new,'a')
    feo = open(filename_e_old,'a')
    fen = open(filename_e_new,'a')
    flog= open(filename_log,'a')
    fcnoise=open(filename_c_noise,'a')
    ##
    colletor = Data_Feeder(dir_prefixes,cutoff = 8,each_system_batch=1) 
    #colletor.collect_files()
    files_in_each_dir = colletor.get_files()

    for idx, file_list in enumerate(files_in_each_dir):
        print(f"{idx+1} :（{dir_prefixes[idx]}）files：")
        for f in file_list:
            print(f"  {f}")

    print("===================================")
    #collletor.Generator_Dataset()
    batch_data_test = colletor.Generate_test_batch(10)
    for (batch, (local_label,local_temp,local_press,local_elements_c,local_coords_c)) in enumerate(batch_data_test):
        tf.print(
                local_label.shape,
                local_temp.shape,
                local_press.shape,
                #local_pred_t.shape,
                local_elements_c.shape,
                #local_elements_n.shape,
                local_coords_c.shape)
                #local_coords_n.shape)
        tf.print(local_coords_c,summarize=500000, output_stream = 'file://'+fco.name)
        #tf.print(local_coords_n,summarize=500000, output_stream = 'file://'+fcn.name)
        tf.print(local_elements_c,summarize=500000, output_stream = 'file://'+feo.name)
        #tf.print(local_elements_n,summarize=500000, output_stream = 'file://'+fen.name)
        A = tf.expand_dims(local_label, axis=-1)
        B = tf.expand_dims(local_temp, axis=-1)
        C = tf.expand_dims(local_press, axis=-1)
        #D = tf.expand_dims(local_pred_t, axis=-1)
        #tf.print(tf.concat([tf.cast(A,dtype=tf.float32),tf.cast(B,dtype=tf.float32),tf.cast(C,dtype=tf.float32),tf.cast(D,dtype=tf.float32)],axis=1),summarize=500000, output_stream = 'file://'+flog.name)
        Noise_creator = create_masks(0.00040,token_noise=0.1)
        (input_tokens,input_coords,noise,iter_i) = Noise_creator.Create_noise(local_elements_c,local_coords_c)
        tf.print("origin_tokens:????",local_elements_c)
        tf.print("input_tokens:?????????????????????",input_tokens)
        E = tf.expand_dims(iter_i, axis=-1)
        tf.print(tf.concat([tf.cast(A,dtype=tf.float32),tf.cast(B,dtype=tf.float32),tf.cast(C,dtype=tf.float32),tf.cast(E,dtype=tf.float32)],axis=1),summarize=500000, output_stream = 'file://'+flog.name)
        tf.print(input_coords,summarize=500000, output_stream = 'file://'+fcnoise.name)
        tf.print("iter_i.shape:",iter_i.shape)
        tf.print(iter_i)
        #tf.print("Pred_t.shape",Pred_t.shape)
        #tf.print(Pred_t)
        #loss_function = loss_1(1,0.8,4,1)
    fco.close()
    fcn.close()
    feo.close()
    fen.close()
    flog.close()
    fcnoise.close()
    #for f in sampled_files:
    #    print(f)
        #print(get_next_filename(f))
    #np_sample = np.array(sampled_files)
    #for f in np_sample:
    #    print(f)

