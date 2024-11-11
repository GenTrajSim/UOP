import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import numpy as np
import glob
import os
import sys
AUTO = tf.data.experimental.AUTOTUNE
from scipy.spatial.distance import pdist,squareform
#tf.set_printoptions(threshold=tf.inf)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_visible_devices(physical_dievices[0],'GPU')
    tf.config.experimental.set_memory_growth(physical_dievices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

dictionary = {'MASK':0, 'C':1, 'O':2, 'CLAS':3} 
# GG_model withiout H, and in the other version with all-atom model

#path_coord = glob.glob('./Data_111/displacement/coord/*.npy') + glob.glob('./Data_111/liq/coord/*.npy') + glob.glob('./Data_111/ice1h/coord/*.npy') + glob.glob('./Data_111/ice1c/coord/*.npy') 
#path_coord = glob.glob('./Data_111/displacement2/coord/13*.npy') + glob.glob('./Data_111/displacement3/coord/13*.npy') # + glob.glob('./Data_111/liq/coord/*.npy')
path_coord = glob.glob('./Data_111/displacement/coord/*.npy') + glob.glob('./Data_111/displacement2/coord/*.npy') + glob.glob('./Data_111/displacement3/coord/*.npy') + glob.glob('./Data_111/liq/coord/*.npy') + glob.glob('./Data_111/displacement3/coord/*.npy') #+ glob.glob('./Data_111/liq/coord/*.npy')
#path_coord = glob.glob('./Data_111/displacement3/coord/*.npy')
#+ glob.glob('./Data_111/MD/coord/*.npy') + glob.glob('./Data_111/ice1h/coord/*.npy') + glob.glob('./Data_111/ice1c/coord/*.npy') ###
MAX_ATOM_N = 10000    ###MAXatom system cutoff
def count_atom_N (path):  ###read_atom Number
    coord =np.load(path)
    atom_N = coord.shape[0]
    return atom_N
A = np.vectorize(count_atom_N)
C = A(path_coord)
path_coord = np.array(path_coord)
path_coord = path_coord[C<=MAX_ATOM_N]

path_test = glob.glob('./Test_Dataset/ice1h/coord/*.npy') + glob.glob('./Test_Dataset/ice1c/coord/*.npy') + glob.glob('./Test_Dataset/ice/coord/*.npy') + glob.glob('./Test_Dataset/hydrate/coord/*.npy') + glob.glob('./Data_111/liq/coord/*.npy')
# + glob.glob('./Data_111/ice1h/coord/*.npy') + glob.glob('./Data_111/ice1c/coord/*.npy') 
#path_test = glob.glob('./Data_111/displacement/coord/*.npy')
#path_test = path_coord
#+ glob.glob('./Test_Dataset/ice6/coord/*.npy') ###

### in this version, the bais (distance) is not be normalized
class Data_Feeder:
    def __init__(self, 
                 total_path, 
                 max_batch_system = 3,
                 max_atoms = 50, 
                 len_dictionary = 4, 
                 d_cut = 5,
                 Training = False):
        self.total_path = total_path
        self.max_batch_system = max_batch_system
        self.max_atoms = max_atoms
        self.len_dictionary = len_dictionary
        self.d_cut = d_cut
        self.Class_id = len_dictionary - 1
        self.Training = Training
    
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
            #sorted_indices = np.argsort(row)[:max_small_index]
            #indices.append(sorted_indices)
        return indices
    
    def local_data (self, indices, data, traget):
        local_dataset = []
        for j in indices[traget]:
            local_dataset.append(data[j,indices[traget]])
        local_dataset = np.reshape(local_dataset,
                                   (local_dataset[-1].shape[0],
                                    local_dataset[-1].shape[0]))
        return local_dataset
    
    def read_npy_file(self, filename, filename2):
        filename = str(filename)
        if self.Training == False :
            filename2 = str(filename2)
        tf.print (filename)
        label = int(filename.split('d/')[1].split('.')[0]) #label
        #data = np.load(filename.numpy().decode())
        data = np.load(filename)
        coord = data[:,1:4]                         #coord
        token = data[:,0].astype(int)+1             #token
        #data2 = np.load(filename2.numpy().decode()) #dist
        if self.Training == False :
            data2 = np.load(filename2) #dist
        else:
            #if label == 0:
            #    tf.print('0000=> ',filename)
            #    noise = np.random.uniform(-0.1, 0.1, size=coord.shape)
            #    coord = coord + noise
            data2 = squareform(pdist(coord,'euclidean'))
        #label = int((filename.numpy().decode('utf-8')).split('d/')[1].split('.')[0]) #label
        #label = int(filename.split('d/')[1].split('.')[0]) #label
        atom_N = data.shape[0]
        #if label == 0:
        #    #noise = np.random.uniform(0.9,1.0)
        #    #d_cut_l = np.random.uniform(7.0,8.0)
        #    #indices = self.indices_distance  (data2,atom_N,(np.random.uniform(7.0,7.5))) # <d_cut index indices
        #    indices = self.indices_distance_d(data2,atom_N,8,np.random.randint(50)) #############  !!!!!!!!!!!!  ###################
        #else:
        #    #indices = self.indices_distance(data2,atom_N,(np.random.uniform(7.0,8.0))) # <d_cut index indices
        #    indices = self.indices_distance_d(data2,atom_N,8,np.random.randint(50)) #############  !!!!!!!!!!!!  ###################
        indices = self.indices_distance_d(data2,atom_N,8,35)#np.random.randint(50)) #############  !!!!!!!!!!!!  ###################
        ############################################################################################################################
        ############!!!!!!!!!!!!!##################
        LOCLA_token = []
        LOCLA_label = []
        LOCLA_atom_edge = []
        LOCLA_coord = []
        LOCLA_distance = []
        #############################################
        for i in range(atom_N):
            atom_N_i = indices[i].shape[0]
            distance = np.zeros((self.max_atoms+1,self.max_atoms+1)) #initial
            dist = self.local_data(indices,data2,i)
            dist = np.pad(dist, ((0, self.max_atoms - atom_N_i), (0, self.max_atoms - atom_N_i)), 'constant')
            distance[1:,1:] = dist
            ###new distance finish
            local_token = token[indices[i]]
            local_token = np.pad(local_token, (0, self.max_atoms - atom_N_i), constant_values=(0))#atom label
            local_token = np.pad(local_token, (1, 0), constant_values=(self.Class_id))#atom label
            ###new token finish
            #centr_coord = coord[i]
            local_coord = coord[indices[i]]
            ################################################ !!! attention 
            #local_coord = local_coord - local_coord.mean(axis=0)
            local_coord = local_coord - coord[i]
            local_coord = np.concatenate([np.zeros((1,3)),
                                          local_coord,
                                          np.zeros((self.max_atoms - atom_N_i,3))], axis=0)
            ### new coord finish
            #coord_dist = np.apply_along_axis(lambda x: np.sqrt((np.sum(x**2))), 1, local_coord )
            #distance[0,:] = coord_dist
            #distance = np.transpose(distance)
            #distance[0,:] = coord_dist
            #Temp
            ###
            atom_edge = (local_token.reshape(-1, 1)*self.len_dictionary) + local_token.reshape(1, -1)
            ###atom_pair finish
            LOCLA_token.append(local_token)
            LOCLA_label.append(label)
            LOCLA_atom_edge.append(atom_edge)
            LOCLA_coord.append(local_coord)
            LOCLA_distance.append(distance)
        #################################################
        LOCLA_token = np.stack(LOCLA_token,axis=0)
        LOCLA_label = np.stack(LOCLA_label,axis=0)
        LOCLA_atom_edge = np.stack(LOCLA_atom_edge,axis=0)
        LOCLA_coord = np.stack(LOCLA_coord,axis=0)
        LOCLA_distance = np.stack(LOCLA_distance,axis=0)
        data = None
        data2 = None
        return (LOCLA_token.astype(np.int32),
                LOCLA_label.astype(np.int32),
                LOCLA_atom_edge.astype(np.int32),
                LOCLA_coord.astype(np.float32),
                LOCLA_distance.astype(np.float32))
    
    def Generator_Dataset(self):
        idn = []
        while (len(idn)<self.max_batch_system):
            path_id = random.randint(0,len(self.total_path)-1)
            if path_id not in idn:
                idn.append(path_id)
        #batch_token = np.empty((0,self.max_atoms+1),dtype=np.int64)
        #batch_label = np.empty((0,),dtype=np.int64)
        #batch_edge =  np.empty((0,self.max_atoms+1,self.max_atoms+1),dtype=np.int64)
        #batch_coord = np.empty((0,self.max_atoms+1,3),dtype=np.float32)
        #batch_dist =  np.empty((0,self.max_atoms+1,self.max_atoms+1),dtype=np.float32)
        #batch_num = 0
        for i in range(len(idn)):
            path = np.array(self.total_path[idn[i]])
            coord_path = np.array(path)
            dist_path = np.char.replace(coord_path,'coord','dist')
            out = self.read_npy_file(coord_path,dist_path)
            #batch_num = batch_num + out[0].shape[0]
            for atom in range(out[0].shape[0]):
                #print (out[0][atom], out[1][atom], out[2][atom], out[3][atom], out[4][atom])
                yield out[0][atom], out[1][atom], out[2][atom], out[3][atom], out[4][atom]

    def Generate_batch(self, batch_size, Repeat_size, shuffle_size):
        dataset = tf.data.Dataset.from_generator(
            self.Generator_Dataset,
            output_types=(tf.int32, tf.int32, tf.int32, tf.float32, tf.float32), 
            output_shapes=((self.max_atoms+1),
                           (),
                           (self.max_atoms+1,self.max_atoms+1),
                           (self.max_atoms+1,3),
                           (self.max_atoms+1,self.max_atoms+1))
        )
        dataset = dataset.repeat(Repeat_size)
        dataset = dataset.shuffle(shuffle_size).batch(batch_size)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def Generate_test_batch(self,batch_size):
        dataset = tf.data.Dataset.from_generator(
                self.Generator_Dataset,
                output_types=(tf.int32, tf.int32, tf.int32, tf.float32, tf.float32),
                output_shapes=(
                    (self.max_atoms+1),
                    (),
                    (self.max_atoms+1,self.max_atoms+1),
                    (self.max_atoms+1,3),
                    (self.max_atoms+1,self.max_atoms+1))
        )
        dataset = dataset.batch(batch_size)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

###  in this version, the bais (distance) is not be normalized
def create_masks_date(token, coord, dist, label, 
                      len_dictionary =4, noise_size = 0.5, noise_size2 = 0.5, 
                      mask_prob = 0.12 , random_token_prob = 0.1, d_cut_off = 5, 
                      Training = False): 
    #input shape: token (bsz, max_atomN), coord(bsz, max_atomN, 3); dist(bsz, max_atomN, max_atomN)
    #print (token)
    #if mask_prob > 0.5:
    #    noise_size = =(1-mask_prob)*(1-mask_prob)
    NO_padding_mask = tf.cast(tf.not_equal(token, 0), dtype=tf.int32)
    NO_padding_clas = tf.cast(tf.not_equal(token, len_dictionary-1), dtype=tf.int32)
    NO_padding = NO_padding_mask * NO_padding_clas
    #print(NO_padding_mask)
    #atom_N = tf.reduce_sum(NO_padding_mask, axis = -1)               ##atom number 
    random_MASK_Prob = tf.random.uniform(shape=NO_padding_mask.shape) < random_token_prob
    
    weight_token = tf.cast((tf.random.uniform(shape=NO_padding.shape))*(len_dictionary-2)+1, dtype=tf.int32) 
    update_token = weight_token*tf.cast(random_MASK_Prob,dtype=tf.int32)*NO_padding
    update_token = tf.where(tf.not_equal(update_token,0), update_token, token)
    ###mask_edge
    mask_edge = (tf.reshape(update_token,[-1,update_token.shape[-1],1])*len_dictionary)+tf.reshape(update_token,[-1,1,update_token.shape[-1]])
    #mask_token = tf.cast(random_MASK_Prob, dtype=tf.int32) * NO_padding_mask
    # add noise atom
    # create noise_coord
    random_MASK_coord_dist_Prob = tf.random.uniform(shape=NO_padding_mask.shape) < mask_prob
    MASK0 = tf.random.uniform(shape=NO_padding_mask.shape) < 1 ########!
    mask_token0= tf.cast(MASK0,dtype=tf.int32) * NO_padding    ########!
    crystal_mask_coord3 = tf.cast(tf.where(tf.not_equal(label,0),1,0),tf.float32)
    crystal_mask_coord3 = tf.tile(tf.expand_dims(crystal_mask_coord3,-1),multiples=[1,131])  ###check
    crystal_mask_coord3 = tf.tile(tf.expand_dims(crystal_mask_coord3,-1),multiples=[1,1,3])  ###check
    #
    mask_token = tf.cast(random_MASK_coord_dist_Prob,dtype=tf.int32) * NO_padding
    #print (mask_token)
    noise = tf.random.uniform(shape=coord.shape,minval=-1,maxval=1)
    ###
    ###liquid_mask
    crystal_mask_coord = tf.cast(tf.where(tf.not_equal(label,0),1,0),tf.float32)
    crystal_mask_coord2= tf.cast(tf.where(tf.not_equal(label,0),0,1),tf.float32) * 0.1
    crystal_mask_coord = tf.tile(tf.expand_dims(crystal_mask_coord,-1),multiples=[1,131])  ###check
    crystal_mask_coord = tf.tile(tf.expand_dims(crystal_mask_coord,-1),multiples=[1,1,3])  ###check
    crystal_mask_coord2= tf.tile(tf.expand_dims(crystal_mask_coord2,-1),multiples=[1,131])  ###check
    crystal_mask_coord2= tf.tile(tf.expand_dims(crystal_mask_coord2,-1),multiples=[1,1,3])  ###check
    #crystal_mask_coord = noise_size*(crystal_mask_coord + crystal_mask_coord2)  + (crystal_mask_coord3 * noise_size2)
    ###
    mask_coord = tf.cast(tf.tile(tf.expand_dims(mask_token,axis = -1), multiples=[1,1,3]), dtype=tf.float32)
    mask_coord0= tf.cast(tf.tile(tf.expand_dims(mask_token0,axis = -1), multiples=[1,1,3]), dtype=tf.float32)
    #
    crystal_mask_coord = (mask_coord*noise_size*(crystal_mask_coord + crystal_mask_coord2))  + (crystal_mask_coord3 * noise_size2 * mask_coord0)
    mask_coord = noise*crystal_mask_coord
    #mask_noise = mask_coord*crystal_mask_coord
    #print (mask_coord)
    mask_coord = mask_coord + coord
    noise = None
    # create nosie_dist
    dist_mask_cnm = tf.tile(tf.expand_dims(NO_padding,axis=-1),multiples=[1,1,tf.shape(NO_padding)[-1]])
    dist_mask_cnm = tf.cast( dist_mask_cnm * (tf.transpose(dist_mask_cnm,perm=[0,2,1])) , dtype = tf.float32)
    if Training == False :
        #mask_dist = squareform(pdist(mask_coord,'euclidean'))
        mask_dist = tf.norm(mask_coord[:,:,None]-mask_coord[:,None,:], axis=-1)
        mask_dist = mask_dist * dist_mask_cnm
    else:
        noise_dist = tf.norm(mask_noise[:,:,None]-mask_noise[:,None,:], axis=-1)
        noise_range = tf.random.uniform(shape=dist.shape,minval=-1,maxval=1)
        noise_range = (noise_range + tf.transpose(noise_range,perm=[0,2,1]))/2
        #noise_dist = tf.random.normal(shape=dist.shape)
        #noise_dist = tf.abs((noise_dist + tf.transpose(noise_dist,perm=[0,2,1]))/2)
        #mask_dist_P = tf.tile(tf.expand_dims(mask_token, axis=-1),multiples=[1,1,mask_token.shape[-1]])
        #mask_dist_P = tf.transpose(mask_dist_P, perm=[0,2,1]) + mask_dist_P
        #mask_dist_P = tf.linalg.set_diag(mask_dist_P, tf.zeros([mask_dist_P.shape[0], mask_dist_P.shape[1]], dtype=tf.int32))
        #mask_dist = noise_dist * tf.cast(mask_dist_P, dtype=tf.float32) * noise_size
        #print (mask_dist)
        mask_dist = (noise_dist*noise_range) + dist
        mask_dist = mask_dist * dist_mask_cnm
    return (
        tf.cast(update_token, dtype=tf.int32),
        tf.cast(mask_edge, dtype=tf.int32),
        tf.cast(mask_coord, dtype=tf.float32),
        tf.cast(mask_dist, dtype=tf.float32)
    )


'''
#test
coord = tf.random.normal(shape=(3,5,3))
dist = tf.random.uniform([3, 5, 5], dtype=tf.float32)
dist = tf.linalg.set_diag(dist, tf.zeros([3, 5], dtype=tf.float32))
token = tf.constant(((0,1,1,1,1),
                     (0,1,1,1,0),
                     (0,1,0,0,0)), dtype=tf.int32)
out = create_masks_date(token, coord, dist)
'''

###  TESTING_DATASET_CREATE

def indices_distance (dist, N_atom, d):
    indices = []
    for i in range(N_atom):
        row = dist[i,:]
        indices.append(np.where(row<d)[0])
    return indices

def local_data (indices, data, traget):
    local_dataset = []
    for j in indices[traget]:
        local_dataset.append(data[j,indices[traget]])
    local_dataset = np.reshape(local_dataset,
                               (local_dataset[-1].shape[0],
                                local_dataset[-1].shape[0]))
    return local_dataset
def read_npy_file(filename, filename2, 
                  len_dictionary = 4, max_atoms = 120, d_cut = 8):
    Class_id = len_dictionary -1
    filename = str(filename)
    filename2 = str(filename2)
    print (filename)
    #data = np.load(filename.numpy().decode())
    data = np.load(filename)
    coord = data[:,1:4]                         #coord
    token = data[:,0].astype(int)+1             #token
    #data2 = np.load(filename2.numpy().decode()) #dist
    data2 = np.load(filename2) #dist
    #label = int((filename.numpy().decode('utf-8')).split('d/')[1].split('.')[0]) #label
    label = int(filename.split('d/')[1].split('.')[0]) #label
    atom_N = data.shape[0]
    indices = indices_distance(data2,atom_N,d_cut) # <d_cut index indices
    LOCLA_token = []
    LOCLA_label = []
    LOCLA_atom_edge = []
    LOCLA_coord = []
    LOCLA_distance = []
    #############################################
    for i in range(atom_N):
        atom_N_i = indices[i].shape[0]
        distance = np.zeros((max_atoms+1,max_atoms+1)) #initial
        dist = local_data(indices,data2,i)
        dist = np.pad(dist, ((0, max_atoms - atom_N_i), (0, max_atoms - atom_N_i)), 'constant')
        distance[1:,1:] = dist
        ###new distance finish
        local_token = token[indices[i]]
        local_token = np.pad(local_token, (0, max_atoms - atom_N_i), constant_values=(0))#atom label
        local_token = np.pad(local_token, (1, 0), constant_values=(Class_id))#atom label
        ###new token finish
        #centr_coord = coord[i]
        local_coord = coord[indices[i]]
        ################################################ !!! attention 
        #local_coord = local_coord - local_coord.mean(axis=0)
        local_coord = local_coord - coord[i]
        local_coord = np.concatenate([np.zeros((1,3)),
                                      local_coord,
                                      np.zeros((max_atoms - atom_N_i,3))], axis=0)
        ### new coord finish
        #coord_dist = np.apply_along_axis(lambda x: np.sqrt((np.sum(x**2))), 1, local_coord )
        #distance[0,:] = coord_dist
        #distance = np.transpose(distance)
        #distance[0,:] = coord_dist
        #Temp
        ###
        atom_edge = (local_token.reshape(-1, 1)*len_dictionary) + local_token.reshape(1, -1)
        ###atom_pair finish
        LOCLA_token.append(local_token)
        LOCLA_label.append(label)
        LOCLA_atom_edge.append(atom_edge)
        LOCLA_coord.append(local_coord)
        LOCLA_distance.append(distance)
    #################################################
    LOCLA_token = np.stack(LOCLA_token,axis=0)
    LOCLA_label = np.stack(LOCLA_label,axis=0)
    LOCLA_atom_edge = np.stack(LOCLA_atom_edge,axis=0)
    LOCLA_coord = np.stack(LOCLA_coord,axis=0)
    LOCLA_distance = np.stack(LOCLA_distance,axis=0)
    data = None
    data2 = None
    return (LOCLA_token.astype(np.int32),
            LOCLA_label.astype(np.int32),
            LOCLA_atom_edge.astype(np.int32),
            LOCLA_coord.astype(np.float32),
            LOCLA_distance.astype(np.float32))

def Gen_Testing_Data(total_path, max_batch_system = 1 , len_dictionary = 4, max_atoms = 120, d_cut = 8):
    idn = []
    while (len(idn)<max_batch_system):
        path_id = random.randint(0,len(total_path)-1)
        if path_id not in idn:
            idn.append(path_id)
    batch_token = np.empty((0,max_atoms+1),dtype=np.int64)
    batch_label = np.empty((0,),dtype=np.int64)
    #batch_label = []
    batch_edge =  np.empty((0,max_atoms+1,max_atoms+1),dtype=np.int64)
    batch_coord = np.empty((0,max_atoms+1,3),dtype=np.float32)
    batch_dist =  np.empty((0,max_atoms+1,max_atoms+1),dtype=np.float32)
    batch_num = 0
    for i in range(len(idn)):
        path = np.array(total_path[idn[i]])
        coord_path = np.array(path)
        dist_path = np.char.replace(coord_path,'coord','dist')
        out = read_npy_file(coord_path,dist_path,len_dictionary = 4, max_atoms = 120, d_cut = 8)
        batch_num = batch_num + out[0].shape[0]
        for atom in range(out[0].shape[0]):
            #print (out[0][atom].shape, out[1][atom].shape, out[2][atom].shape, out[3][atom].shape, out[4][atom].shape)
            #yield out[0][atom], out[1][atom], out[2][atom], out[3][atom], out[4][atom]
            batch_token = np.concatenate([batch_token, np.expand_dims(out[0][atom],axis=0)], axis=0)
            batch_label = np.concatenate([batch_label, np.expand_dims(out[1][atom],axis=0)], axis=0)
            batch_edge = np.concatenate([batch_edge, np.expand_dims(out[2][atom],axis=0)], axis=0)
            batch_coord = np.concatenate([batch_coord, np.expand_dims(out[3][atom],axis=0)], axis=0)
            batch_dist = np.concatenate([batch_dist, np.expand_dims(out[4][atom],axis=0)], axis=0)
    return (
        #batch_token,
        #batch_label,
        #batch_edge,
        #batch_coord,
        #batch_dist
        tf.convert_to_tensor(batch_token, dtype = tf.int32),
        tf.convert_to_tensor(batch_label, dtype = tf.int32),
        tf.convert_to_tensor(batch_edge, dtype = tf.int32),
        tf.convert_to_tensor(batch_coord, dtype = tf.float32),
        tf.convert_to_tensor(batch_dist, dtype = tf.float32)
    )

'''
out = Gen_Testing_Data(path_coord, max_batch_system = 1 , len_dictionary = 4, max_atoms = 120, d_cut = 8)
out
'''

'''
#test
coord = tf.random.normal(shape=(3,5,3))
dist = tf.random.uniform([3, 5, 5], dtype=tf.float32)
dist = tf.linalg.set_diag(dist, tf.zeros([3, 5], dtype=tf.float32))
token = tf.constant(((0,1,1,1,1),
                     (0,1,1,1,0),
                     (0,1,0,0,0)), dtype=tf.int32)
out = create_masks_date(token, coord, dist)
'''
class SelfMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 dropout=0.1, 
                 bias=True, 
                 scaling_factor=1): #scaling_factor -> dk's factor
        super(SelfMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        assert embed_dim % self.num_heads == 0
        self.head_dim = embed_dim // self.num_heads
        self.dk = tf.cast(self.head_dim, tf.float32)
        self.scaling = 1/tf.sqrt(self.dk * scaling_factor)
        #self.in_proj = tf.keras.layers.Dense(embed_dim*3,use_bias=bias) #embeding: embed_dim->embed_dim*3
        #self.out_proj = tf.keras.layers.Dense(embed_dim,use_bias=bias) #embeding: embed_dim->embed_dim
        ###
        self.wq = tf.keras.layers.Dense(embed_dim,use_bias=bias)
        self.wk = tf.keras.layers.Dense(embed_dim,use_bias=bias)
        self.wv = tf.keras.layers.Dense(embed_dim,use_bias=bias)
        #self.dense = tf.keras.layers.Dense(d_model)         #use 'final' dense layer ???
        #self.linear_bias = tf.keras.layers.Dense(num_heads) #pair (depth -> num_head)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.softmax1 = tf.keras.layers.Softmax(axis=-1)
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias)
        #self.dk = tf.cast(self.depth, tf.float32)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        #num_heads * depth = d_model
        x = tf.transpose(x, perm=[0,2,1,3])
        x = tf.reshape(x, (batch_size*self.num_heads, -1, self.head_dim))
        return x #batch_size * num_heads, seq, depth
    
    def call(self, 
             query,          #shape (bsz, seq_len, embed_dim)
             key,            #shape (bsz, seq_len, embed_dim)
             value,           #shape (bsz, seq_len, embed_dim)
             key_padding_mask, #shape (bsz, seq_len)
             attn_bias,       #shape (bsz * num_head, Natom, Natom) ps. Natom == seq_len
             training = False,
             return_attn=True):
        bsz = tf.shape(query)[0]
        tgt_len = tf.shape(query)[1]
        embed_dim = tf.shape(query)[2]
        #assert embed_dim == self.embed_dim
        #
        #batch_size = tf.shape(q)[0]
        q = self.wq(query)
        q = q*self.scaling
        k = self.wq(key)
        v = self.wq(value)
        q = self.split_heads(q, bsz)
        k = self.split_heads(k, bsz)
        v = self.split_heads(v, bsz)
        src_len = tf.shape(k)[1]################## seq (in my opinion:  tgt_len != src_len)
        #                                          check  tgt_len = src_len + len(classify_token) ?
        if key_padding_mask is not None and tf.rank(key_padding_mask)==0: ### check grammar
            key_padding_mask = None

        if key_padding_mask is not None: #shape: bsz,src_len
            assert tf.shape(key_padding_mask)[0] == bsz
            assert tf.shape(key_padding_mask)[1] == src_len

        attn_weights = tf.matmul(q,k,transpose_b=True) #matmul_qk
        #assert list(tf.shape(attn_weights)) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))
            key_padding_mask = tf.expand_dims(key_padding_mask, axis=1)
            key_padding_mask = tf.expand_dims(key_padding_mask, axis=2)
            key_padding_mask = tf.cast(key_padding_mask, tf.bool)
            attn_weights = tf.where(key_padding_mask, float("-inf"), attn_weights)
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))
        '''bias_shape need to be the same as attn_weights (bsz, self.num_heads, tgt_len, src_len)
            tgt_len invlove classify token + src_len
            tgt_len == src_len
            check_bias_shape!!!!!!!!!!!'''
        if attn_bias is not None:
            attn_bias = tf.reshape(attn_bias, (bsz*self.num_heads, tgt_len, src_len)) #!!check shape
            attn_weights += attn_bias
        attn = self.softmax1(attn_weights)
        attn = self.dropout1(attn, training=training)
        o = tf.matmul(attn, v)
        #assert list(tf.shape(o)) == [bsz * self.num_heads, tgt_len, self.head_dim]
        o = tf.reshape(o, (bsz, self.num_heads, tgt_len, self.head_dim))
        o = tf.transpose(o, perm=[0,2,1,3])
        o = tf.reshape(o, (bsz, tgt_len, embed_dim))
        o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn
        #matmul_qk=tf.matmul(q,k,transpose_b=True)
        #bias = self.linear_bias(q_pair)
        #bias = tf.transpose(bias, perm=[0,3,1,2])
        #if mask is not None:##############can del
        #    matmul_qk+=(mask * -1e9)######can del
        #attention_weights = matmul_qk + bias
        #attention_weights = self.dropout1(attention_weights, training=training)
        #attention_weights = self.softmax1(attention_weights)
        #output = tf.matmul(attention_weights,v)
        #output = tf.transpose(output, perm=[0,2,1,3])
        #output = tf.reshape(output,(batch_size, -1, self.d_model))
        #output = self.dense(output)
        #return output, attention_weights
'''
 Herein, need to check the shapes of key_padding_mask and attn_bias !!!!!!!
'''


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, 
                embed_dim = 768, # can/need smaller
                ffn_embed_dim = 3072, # can/need smaller
                attention_heads = 8,
                dropout = 0.1,
                attention_dropout = 0.1,
                activation_dropout = 0.0,
                #activation_fn = "gelu",
                post_ln = False):
        super(TransformerEncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        
        self.attention_dropout = attention_dropout
        #self.dropout = dropout
        #self.activation_dropout = activation_dropout

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(activation_dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        
        self.activation_fn = tf.keras.layers.Activation('gelu')
        
        self.self_attn = SelfMultiHeadAttention(
            self.embed_dim,
            attention_heads,
            dropout = attention_dropout
        )
        
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.fc1 = tf.keras.layers.Dense(ffn_embed_dim) # embed_dim-> ffn_embed_dim
        self.fc2 = tf.keras.layers.Dense(self.embed_dim) # ffn_embed_dim->embed_dim
        self.final_layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.post_ln = post_ln
    def call (self, 
              x, 
              attn_bias, 
              padding_mask, 
              training = False,
              return_attn = False):
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query = x, 
            key = x,
            value = x,
            key_padding_mask = padding_mask, 
            attn_bias = attn_bias,
            training = training,
            return_attn = return_attn
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = self.dropout1(x, training=training)
        x = x + residual
        if self.post_ln:
            x = self.self_attn_layer_norm(x)
        residual = x
        
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout2(x, training=training)
        x = self.fc2(x)
        x = self.dropout3(x, training=training)
        x = x + residual
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs

@tf.function
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return tf.math.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(tf.keras.layers.Layer):
    def __init__(self, K=128, edge_types=1024):
        super(GaussianLayer, self).__init__()
        self.K = K
        self.means = tf.keras.layers.Embedding(1, K)
        self.stds = tf.keras.layers.Embedding(1, K)
        self.mul = tf.keras.layers.Embedding(edge_types, 1)
        self.bias = tf.keras.layers.Embedding(edge_types, 1)
        self.means.build((None,))
        self.stds.build((None,))
        self.mul.build((None,))
        self.bias.build((None,))
        self.means.set_weights([tf.keras.initializers.RandomUniform(0, 3)(self.means.weights[0].shape)])
        self.stds.set_weights([tf.keras.initializers.RandomUniform(0, 3)(self.stds.weights[0].shape)])
        self.bias.set_weights([tf.keras.initializers.Constant(0)(self.bias.weights[0].shape)])
        self.mul.set_weights([tf.keras.initializers.Constant(1)(self.mul.weights[0].shape)])
        
    def call(self, x, edge_type):
        mul = self.mul(edge_type)###type_as(x)
        bias = self.bias(edge_type)###type_as(x)
        x = mul * (tf.expand_dims(x, axis=-1)) + bias
        x = tf.tile(x, [1, 1, 1, self.K])
        mean = tf.reshape(self.means.weights[0], [-1])
        std = tf.math.abs(tf.reshape(self.stds.weights[0], [-1])) + 1e-5
        return gaussian(x, mean, std)
        '''
        input-> x:(bsz, N, N) 
                edge_type:(bsz, N, N) 
                (e.g. et = tf.constant([[0,1,2,3,4,5,6,7,8,9,0],[0,2,3,1,1,1,1,3,8,5,0]])
                  et_edge_type = tf.expand_dims(et,-1)*10 + tf.expand_dims(et,-2))
        output-> (bsz, N, N, K)
        '''

class TransformerEncoderWithPair(tf.keras.layers.Layer):
    def __init__(self, 
                 encoder_layers = 6, 
                 embed_dim = 768, 
                 ffn_embed_dim = 3072, 
                 attention_heads = 8, 
                 emb_dropout = 0.1, 
                 dropout = 0.1, 
                 attention_dropout = 0.1, 
                 activation_dropout = 0.0, 
                 max_seq_len = 256, #???
                 # activation_fn = "gelu", 
                 post_ln = False, 
                 no_final_head_layer_norm = False):
        super(TransformerEncoderWithPair, self).__init__()
        self.encoder_layers = encoder_layers
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len ###????
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        if not post_ln:
            self.final_layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        else:
            self.final_layer_norm = None
        if not no_final_head_layer_norm:
            self.final_head_layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        else:
            self.final_head_layer_norm = None
        self.layers = [TransformerEncoderLayer(embed_dim = self.embed_dim,
                                               ffn_embed_dim = ffn_embed_dim,
                                               attention_heads = attention_heads,
                                               dropout = dropout,
                                               attention_dropout = attention_dropout,
                                               activation_dropout = activation_dropout,
                                               #activation_fn = "gelu",
                                               post_ln = post_ln) 
                       for _ in range(encoder_layers)]
        self.Dropout_emb = tf.keras.layers.Dropout(emb_dropout)
        ##
    def call (self, emb, attn_mask, padding_mask, training = False):
        bsz = tf.shape(emb)[0]
        seq_len = tf.shape(emb)[1]
        x = self.emb_layer_norm(emb)
        x = self.Dropout_emb(x, training=training)
        if padding_mask is not None:
            x = x * (1-tf.cast(tf.expand_dims(padding_mask, axis=-1), x.dtype))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask
        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                attn_mask = tf.reshape(attn_mask, (tf.shape(x)[0], -1, seq_len, seq_len)) #check shape
                padding_mask = tf.expand_dims(padding_mask, axis=1)
                padding_mask = tf.expand_dims(padding_mask, axis=2)
                padding_mask = tf.cast(padding_mask, tf.bool)
                attn_mask = tf.where(padding_mask, tf.cast(fill_val, dtype=tf.float32), attn_mask)
                attn_mask = tf.reshape(attn_mask, (-1, seq_len, seq_len))
                padding_mask = None
            return attn_mask, padding_mask
        assert attn_mask is not None
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)
        for i in range(self.encoder_layers):
            x, attn_mask, _ = self.layers[i](
                x, padding_mask=padding_mask, attn_bias=attn_mask, training = training, return_attn=True
            )
        def norm_loss(x, eps=1e-10, tolerance=1.0):
            x = tf.cast(x, tf.float32)
            max_norm = tf.sqrt(tf.cast((tf.shape(x)[-1]),tf.float32))
            norm = tf.sqrt(tf.reduce_sum(x**2, -1) + eps)
            error = tf.nn.relu(tf.abs(norm - max_norm)-tolerance)
            return error
        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return tf.reduce_mean(
                (tf.reduce_sum(mask*value, dim) / ( eps + tf.reduce_sum(mask, dim))), -1
            )
        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - tf.cast(input_padding_mask, tf.float32)
        else:
            token_mask = tf.ones_like(x_norm)
        x_norm = masked_mean(token_mask, x_norm)
        if self.final_layer_norm is not None:
            x = self.final_layer_norm (x)
        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask , 0)
        #attn_mask = (
        attn_mask = tf.reshape(attn_mask, (bsz, -1, seq_len, seq_len))
        attn_mask = tf.transpose(attn_mask, perm=[0,2,3,1])
        #)
        #delta_pair_repr = (
        delta_pair_repr = tf.reshape(delta_pair_repr, (bsz, -1, seq_len, seq_len))
        delta_pair_repr = tf.transpose(delta_pair_repr, perm=[0,2,3,1])
        #)
        #????nuclear funtion following#
        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        #[bsz, seq_len, 1] * [bsz, 1, seq_len] -> shape [bsz, seq_len, seq_len]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(pair_mask, delta_pair_repr_norm, dim=(-1, -2))
        # below #
        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)
        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm

'''
TransformerEncoderWithPair input -> 
emb: bsz, seq_len, embided_dim
attn_mask: bsz * head_num, seq_len, seq_len
padding_mask: bsz, seq_len (e.g. [[T.,F.,F.,F.,T.],[F.,F.,F.,T.,T.],...])
training: Trure
Output ->
x: bsz, seq_len, embided_dim
attn_mask: bsz, seq_len, seq_len, head_num (e.g. [[[[-inf,-inf,...],[n,n,n,...],[],...[head_num]],[],...],[],...])
delta_pair_repr: bsz, seq_len, seq_len, head_num (e.g. [[[[0,0,...],[n,n,n,...],[],...[head_num]],[],...],[],...])
x_norm: a score
delta_pair_repr_norm: a score
'''

class MaskLMHead(tf.keras.layers.Layer):
    def __init__(self, 
                 embed_dim, 
                 output_dim, 
                 #activation_fn = 'gelu', 
                 weight=None):
        super(MaskLMHead, self).__init__()
        self.dense = tf.keras.layers.Dense(embed_dim)
        self.activation_fn = tf.keras.layers.Activation('gelu')
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        if weight is None:
            #weight = tf.keras.layers.Dense(output_dim, use_bias=False).weights[0]
            # or can test
            weight = tf.keras.layers.Dense(output_dim, use_bias=False)
            weight.build((None,embed_dim))
        self.weight = weight.weights[0]
        #self.bias = tf.zeros_like(output_dim)
        self.bias = tf.Variable(tf.zeros(output_dim))
    def call (self, features, masked_tokens=None):
        if masked_tokens is not None:
            features = tf.gather(features, masked_tokens)
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        #print("self.weight = ")
        #print(tf.shape(self.weight))
        #print("self.bias = ")
        #print(self.bias)
        x = tf.matmul(x, self.weight) + self.bias
        #x = self.weight(x) + self.bias
        return x

class ClassificationHead(tf.keras.layers.Layer):
    def __init__(
        self, 
        input_dim, 
        inner_dim, 
        num_classes, 
        #activation_fn, 
        pooler_dropout
    ):
        super(ClassificationHead, self).__init__()
        self.dense = tf.keras.layers.Dense(inner_dim)
        #self.activation_fn = tf.keras.activations.tanh() #### need check
        self.activation_fn = tf.keras.layers.Activation('gelu')
        self.dropout = tf.keras.layers.Dropout(pooler_dropout)
        self.out_proj = tf.keras.layers.Dense(num_classes)
    def call (self, features):
        x = features[:, 0, :] 
        x = self.dropout(x)
        x = self.dense(x)
        #x = tf.keras.activations.tanh(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class NonLinearHead(tf.keras.layers.Layer):
    def __init__(
        self, 
        input_dim,
        out_dim,
        #activation_fn,
        hidden = None
    ):
        super(NonLinearHead, self).__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = tf.keras.layers.Dense(hidden)
        self.linear2 = tf.keras.layers.Dense(out_dim)
        self.activation_fn = tf.keras.layers.Activation('gelu')
    def call (self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

class DistanceHead(tf.keras.layers.Layer):
    def __init__(
        self,
        heads,
        #activation_fn
    ):
        super(DistanceHead, self).__init__()
        self.dense = tf.keras.layers.Dense(heads)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.out_proj = tf.keras.layers.Dense(1)
        self.activation_fn =  tf.keras.layers.Activation('gelu')
    def call (self, x):
        #bsz, seq_len, seq_len, _ = tf.shape(x)
        bsz = x.shape[0]
        seq_len = x.shape[1]
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x)
        x = tf.reshape(x, (bsz, seq_len, seq_len))
        x = ( x + tf.transpose(x, perm=[0, 2, 1]) ) * 0.5
        return x
'''
-------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------MAIN--------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------
'''
class Gen3Dmol_Classify(tf.keras.layers.Layer):
    def __init__(
        self, 
        encoder_layers = 15,
        encoder_embed_dim = 512,
        encoder_ffn_embed_dim = 2048,
        encoder_attention_heads = 32,
        dropout = 0.1,
        emb_dropout = 0.1,
        attention_dropout = 0.1,
        activation_dropout = 0.0,
        pooler_dropout = 0.0,
        max_seq_len = 512,
        #activation_fn = "rel
        #pooler_activation_fn = "tanh",
        post_ln = False,
        masked_token_loss = -1.0,
        masked_token_pred = 1.0,
        masked_coord_loss = -1.0,
        masked_dist_loss = -1.0,
        x_norm_loss = -1.0,
        delta_pair_repr_norm_loss = -1.0,
        num_classes = 3,
        crytal_class = 10,
        dictionary = None
    ):
        super(Gen3Dmol_Classify, self).__init__()
        #self.padding_idx = dictionary['MASK']
        self.padding_idx = 0 
        ##
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.pooler_dropout =  pooler_dropout
        self.max_seq_len = max_seq_len
        #activation_fn = "rel
        #pooler_activation_fn = "tanh",
        self.post_ln = post_ln
        self.masked_token_loss = masked_token_loss
        self.masked_token_pred = masked_token_pred
        self.masked_coord_loss = masked_coord_loss
        self.masked_dist_loss = masked_dist_loss
        self.x_norm_loss =x_norm_loss                        ############
        self.delta_pair_repr_norm_loss = delta_pair_repr_norm_loss#######
        self.num_classes = num_classes
        self.crytal_class = crytal_class
        ##
        self.embed_tokens = tf.keras.layers.Embedding(
            len(dictionary), self.encoder_embed_dim, mask_zero=True
        )
        self.encoder = TransformerEncoderWithPair(
            encoder_layers = self.encoder_layers, 
            embed_dim = self.encoder_embed_dim, 
            ffn_embed_dim = self.encoder_ffn_embed_dim, 
            attention_heads = self.encoder_attention_heads, 
            emb_dropout = self.emb_dropout, 
            dropout = self.dropout, 
            attention_dropout = self.attention_dropout, 
            activation_dropout = self.activation_dropout, 
            max_seq_len = self.max_seq_len, #???
            # activation_fn = "gel
            post_ln = False, 
            no_final_head_layer_norm = self.delta_pair_repr_norm_loss
        )
        if self.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim = self.encoder_embed_dim,
                output_dim = self.crytal_class,
                weight=None
            )
        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, self.encoder_attention_heads
        )
        self.gbf = GaussianLayer(K, n_edge_type)
        if self.masked_coord_loss > 0:
            self.pair2coord_proj  = NonLinearHead(
                self.encoder_attention_heads, 1
            )
        if self.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                self.encoder_attention_heads
            )
        self.classification_heads = ClassificationHead(input_dim = self.encoder_embed_dim,
                                                       inner_dim = self.encoder_embed_dim,
                                                       num_classes = self.num_classes, 
                                                       #activation_fn, 
                                                       pooler_dropout = self.pooler_dropout)
    # classmmethod
    # def build_model(cls, args, task): 
    # return cls(args, task.dictionary)
    def call(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,   # always None
        Not_only_features=False,
        training = False
        #classification_head_name=True
    ):
        #if not classification_head_name:
        #    features_only = True
        padding_mask = tf.equal(src_tokens, self.padding_idx)
        #if not tf.reduce_any(padding_mask):
        #    padding_mask = None
        x = self.embed_tokens(src_tokens)
        def get_dist_features(dist, et):
            n_node = dist.shape[-1]
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = tf.transpose(graph_attn_bias, perm=[0,3,1,2])
            graph_attn_bias = tf.reshape(graph_attn_bias, (-1, n_node, n_node))
            return graph_attn_bias
        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias,training = training)
        #encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        encoder_pair_rep = tf.where(tf.math.is_finite(encoder_pair_rep), encoder_pair_rep, tf.cast(tf.zeros_like(encoder_pair_rep),dtype=tf.float32))
        encoder_distance = None
        encoder_coord = None
        #if not features_only:
        if self.masked_token_loss > 0:
            logits = self.lm_head(encoder_rep, encoder_masked_tokens)
        if self.masked_coord_loss > 0:
            coords_emb = src_coord
            if padding_mask is not None:
                atom_num = tf.reshape((tf.reduce_sum( 1 - tf.cast(padding_mask, x.dtype) , axis=1) -1),
                                      (-1, 1, 1, 1))
            else:
                atom_num = tf.shape(src_coord, 1) - 1 ###?????check!!!!!!!!!!!!!!!!!!!!!!!!!!!!(-1)?????
            delta_pos = tf.expand_dims(coords_emb, axis=1) - tf.expand_dims(coords_emb, axis=2)
            attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
            coord_update = delta_pos / atom_num * attn_probs
            coord_update = tf.reduce_sum(coord_update, axis=2)
            encoder_coord = coords_emb + coord_update
        if self.masked_dist_loss > 0:
            encoder_distance = self.dist_head(encoder_pair_rep)
        if Not_only_features:
            logits_h = self.classification_heads(encoder_rep)
            return (
                logits,   ###token type
                logits_h,  ##crystal type
                encoder_distance,
                encoder_coord,
                x_norm,
                delta_encoder_pair_rep_norm
                )
        else:
            return (
                logits,
                encoder_distance,
                encoder_coord,
                x_norm,
                delta_encoder_pair_rep_norm
            )


'''
test:

test_layer = Gen3Dmol_Classify(
    masked_token_loss = 1.0,
    masked_token_pred = 1.0,
    masked_coord_loss = 1.0,
    masked_dist_loss = 1.0,
    x_norm_loss = 1.0,
    delta_pair_repr_norm_loss = 1.0,
    dictionary=dictionary)

y = test_layer(token,
               distance,
               coord,
               edge_type,
               encoder_masked_tokens=None,   # always None
               Not_only_features=True)
'''

model = Gen3Dmol_Classify(
    masked_token_loss = 1.0,
    masked_token_pred = 1.0,
    masked_coord_loss = 1.0,
    masked_dist_loss = 1.0,
    x_norm_loss = 1.0,
    delta_pair_repr_norm_loss = 1.0,
    dictionary=dictionary,
    crytal_class = 4,###token number
    num_classes = 31, ###state number (pre-transition and post-transition) 
    encoder_layers = 15,
    encoder_embed_dim =  512, #256, ###so big maybe 256 or 128 or 64 or 32
    encoder_ffn_embed_dim = 2048, ### maybe small
)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)
loss_object_cry = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss_function_1(pred_token, orign_token, 
                    pred_crystal, real_crystal, 
                    new_coord, real_coord, 
                    new_dist, real_dist, 
                    loss_x_norm, loss_delta_encoder_pair_rep_norm, 
                    len_dictionary =4,
                    masked_token_loss = 1.0, crysral_class_loss =1.0,
                    masked_coord_loss = 1.0, masked_dist_loss =1.0,
                    x_norm_loss =1.0, delta_pair_repr_norm_loss =1.0):
    #mask_token = tf.cast(tf.not_equal(orign_token, 0), dtype=tf.int32)
    mask_token = tf.cast(tf.where(tf.not_equal(orign_token, 0)&tf.not_equal(orign_token,3),1,0), dtype=tf.int32) #####   3->classify need change in different object
    ##########!!!
    pred_t_One = tf.math.argmax(tf.nn.log_softmax(pred_token, axis=-1),-1)
    mask_tokenP= tf.cast(tf.where(tf.not_equal(pred_t_One , 0)&tf.not_equal(pred_t_One ,3),1,0), dtype=tf.int32) #####   3->classify need change in different object
    ##########!!!
    mask_token_classify = tf.cast(tf.not_equal(orign_token, 0), dtype=tf.int32)
    #NO_padding_clas = tf.cast(tf.not_equal(orign_token, len_dictionary-1), dtype=tf.int32)
    #mask_token = mask_token * NO_padding_clas
    #sample_size = tf.reduce_sum(mask_token, axis = -1)  
    orign_token = orign_token * mask_token_classify
    #pred_token = pred_token*NO_padding_clas
    token_loss = loss_object(orign_token,tf.nn.log_softmax(pred_token,axis=-1))
    ####
    crystal_loss = loss_object_cry(real_crystal, tf.nn.log_softmax(pred_crystal,axis=-1))
    ##liquid_mask
    crystal_mask_coord = tf.cast(tf.where(tf.not_equal(real_crystal,0),1,0),tf.float32) 
    crystal_mask_coord2= tf.cast(tf.where(tf.not_equal(real_crystal,0),0,1),tf.float32) *0.000001 ## 0.0000001
    crystal_mask_coord = tf.tile(tf.expand_dims(crystal_mask_coord,-1),multiples=[1,131])  ###check
    crystal_mask_coord = tf.tile(tf.expand_dims(crystal_mask_coord,-1),multiples=[1,1,3])  ###check
    crystal_mask_coord2= tf.tile(tf.expand_dims(crystal_mask_coord2,-1),multiples=[1,131])  ###check
    crystal_mask_coord2= tf.tile(tf.expand_dims(crystal_mask_coord2,-1),multiples=[1,1,3])  ###check
    crystal_mask_coord = crystal_mask_coord + crystal_mask_coord2
    ####
    crystal_mask_dist  = tf.cast(tf.where(tf.not_equal(real_crystal,0),1,0),tf.float32) 
    crystal_mask_dist2 = tf.cast(tf.where(tf.not_equal(real_crystal,0),0,1),tf.float32) *0.000001 ## 0.0000001
    crystal_mask_dist  = tf.tile(tf.expand_dims(crystal_mask_dist,-1),multiples=[1,131]) ###check
    crystal_mask_dist  = tf.tile(tf.expand_dims(crystal_mask_dist,-1),multiples=[1,1,131]) ###check
    crystal_mask_dist2 = tf.tile(tf.expand_dims(crystal_mask_dist2,-1),multiples=[1,131]) ###check
    crystal_mask_dist2 = tf.tile(tf.expand_dims(crystal_mask_dist2,-1),multiples=[1,1,131]) ###check
    crystal_mask_dist  = crystal_mask_dist + crystal_mask_dist2
    #crystal_mask_dist  = tf.tile(tf.expand_dims(crystal_mask_dist, -1),multiples=[1,tf.shape(real_dist)[-3],1])
    #crystal_mask_dist  = tf.tile(tf.expand_dims(crystal_mask_dist, -1),multiples=[1,1,tf.shape(real_dist)[-1],tf.shape(real_dist)[-1]])
    ####
    if new_coord is not None:
        coord_loss = tf.compat.v1.losses.huber_loss(
            labels = real_coord * crystal_mask_coord,
            predictions = new_coord * crystal_mask_coord,
            weights = tf.cast(tf.tile(tf.expand_dims(mask_tokenP,axis = -1), multiples=[1,1,3]), dtype=tf.float32),
            delta = 0.01
            #reduction=tf.keras.losses.Reduction.NONE
        )
    else:
        coord_loss = 0
    ###
    if new_dist is not None:
        dist_MASK = tf.tile(tf.expand_dims(mask_tokenP, axis=-1),
                            multiples=[1,1,tf.shape(mask_tokenP)[-1]])
        dist_MASK = dist_MASK * (tf.transpose(dist_MASK, perm=[0,2,1]))
        ####whether add diagonal == 0
        #dist_MASK = tf.linalg.set_diag(dist_MASK, tf.zeros([batch_size, 5], dtype=tf.float32))
        dist_loss = tf.compat.v1.losses.huber_loss(
            labels = real_dist * crystal_mask_dist,
            predictions = new_dist  * crystal_mask_dist,
            weights = tf.cast(dist_MASK, dtype=tf.float32),
            delta = 0.01
        )
    if loss_x_norm is not None:
        norm_loss = loss_x_norm
    if loss_delta_encoder_pair_rep_norm is not None:
        pair_loss = loss_delta_encoder_pair_rep_norm
    loss = ((token_loss*masked_token_loss)+
            (crystal_loss*crysral_class_loss)+
            (coord_loss*masked_coord_loss)+
            (dist_loss*masked_dist_loss)+
            (norm_loss*x_norm_loss)+
            (pair_loss*delta_pair_repr_norm_loss))
    return loss, token_loss, crystal_loss, dist_loss, coord_loss


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=100000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
    def __call__(self, step):
        arg1 = tf.math.rsqrt(tf.cast(step, dtype=tf.float32))
        arg2 = tf.cast(step,dtype=tf.float32) * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)  

l_r = CustomSchedule(512)

class CustomSchedule2(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=1000000):
        super(CustomSchedule2, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
    def __call__(self, step):
        arg = tf.math.rsqrt(tf.cast(step + self.warmup_steps, dtype=tf.float32))
        return tf.math.rsqrt(self.d_model) * arg

#global_step.assign(0)                                                                                      ##
#l_r2 = CustomSchedule2(512)                                                                                ##
#optimizer = tf.keras.optimizers.Adam(learning_rate=l_r2, beta_1=0.9, beta_2=0.98,
#                                             epsilon=1e-9)

optimizer = tf.keras.optimizers.Adam(learning_rate=l_r, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

checkpoint_path = "./checkpoint_COORD_1.00_vDAMN_ln_liq20/train"
#save_train_log = open('./Gen3DTransformer/log/training.log', 'w')
step = tf.Variable(0, name="step")
ckpt = tf.train.Checkpoint(model, optimizer=optimizer, step=step)
#train_log = tf.logging.FileHandler('train.log')     #write logging
#tf.logging.set_verbosity(tf.logging.INFO)           #
#tf.logging.getLogger().addHandler(train_log)        #write logging
#step = tf.Variable(0, name="step")
#checkpoint = tf.train.Checkpoint(step=step)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if checkpoint exit, read checkpoint
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')
    

train_total_loss = tf.keras.metrics.Mean(name='train_total_loss')
train_token_loss = tf.keras.metrics.Mean(name='train_token_loss')
train_crystal_loss = tf.keras.metrics.Mean(name='train_crystal_loss')
train_dist_loss = tf.keras.metrics.Mean(name='train_dist_loss')
train_coord_loss = tf.keras.metrics.Mean(name='train_coord_loss')
train_accuracy_label = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
#train_accuracy_token = tf.keras.metrics.SparseCategoricalAccuracy(
#    name='train_accuracy')

@tf.function
def train_step(orign_token, label, edge, coord, dist, len_dictionary):
    noise_size_set = random.uniform(0.65,0.8)
    #noise_size_set = 0.6
    (
        mask_token, 
        mask_edge,
        mask_coord, 
        mask_dist
    ) = create_masks_date(
        orign_token, coord, dist, label,
        len_dictionary =len_dictionary, noise_size = noise_size_set, noise_size2 = random.uniform(0.3,0.5),  ################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #nosie_size_set = noise_size,
        #mask_prob = random.uniform(0.2,0.24) , 
        mask_prob = random.uniform(0.4,0.8), #1 ,
        #random_token_prob = random.uniform(0,0.3)
        random_token_prob = random.uniform(0.0,0.4)
        , Training = False ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!wo bu lijie , i think it should be False
    )
    with tf.GradientTape() as tape:
        (
            pred_token,pred_crystal,pred_dist,pred_coord,x_norm,pair_norm
        ) = model(mask_token,mask_dist,mask_coord,mask_edge,Not_only_features=True,training = True) ##maybe use edge, not mask_edge
        #print (pred_dist)
        loss, token_loss, crystal_loss, dist_loss, coord_loss = loss_function_1(
            pred_token, orign_token, 
            pred_crystal, label,
            #None,coord,
            pred_coord,coord,
            pred_dist,dist,
            x_norm,pair_norm,
            len_dictionary =len_dictionary,
            masked_token_loss = 1, crysral_class_loss =1,
            masked_coord_loss = 10000, masked_dist_loss = 10000,
            x_norm_loss =1, delta_pair_repr_norm_loss = 1
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_total_loss(loss)
    train_token_loss(token_loss)
    train_crystal_loss(crystal_loss)
    train_dist_loss(dist_loss)
    train_coord_loss(coord_loss)
    train_accuracy_label(label, pred_crystal)

@tf.function
def test_step(orign_token, label, edge, coord, dist, len_dictionary, epoch, noise_size_input):
    noise_size_set = noise_size_input
    #tf.print("noise_size: ",noise_size_set)
    (
        mask_token,
        mask_edge,
        mask_coord,
        mask_dist
    ) = create_masks_date(
        orign_token, coord, dist, label,
        len_dictionary =len_dictionary, noise_size = noise_size_set, noise_size2 = 0.5, #############################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        mask_prob = 0.25 , random_token_prob = 0.
        , Training = False
        #learn lable should be close
    )
    (
        pred_token,pred_crystal,pred_dist,pred_coord,x_norm,pair_norm
    ) = model(orign_token,dist,coord,edge,Not_only_features=True, training = False) ##maybe use edge, not mask_edge
    # model(mask_token,mask_dist,mask_coord,mask_edge,Not_only_features=True,training = False) ##maybe use edge, not mask_edge
    #= model(orign_token,dist,coord,edge,Not_only_features=True, training = False) ##maybe use edge, not mask_edge
    loss, token_loss, crystal_loss, dist_loss, coord_loss = loss_function_1(
        pred_token, orign_token,
        pred_crystal, label,
        pred_coord,coord,
        pred_dist,dist,
        x_norm,pair_norm,
        len_dictionary =len_dictionary,
        masked_token_loss = 1, crysral_class_loss =1,
        masked_coord_loss = 1, masked_dist_loss =1,
        x_norm_loss =1, delta_pair_repr_norm_loss =1
    )
    test_total_loss(loss)
    test_token_loss(token_loss)
    test_crystal_loss(crystal_loss)
    test_dist_loss(dist_loss)
    test_coord_loss(coord_loss)
    test_accuracy_label(label, pred_crystal)
    No_padding_token = tf.cast(tf.not_equal(orign_token, 0), dtype=tf.int32)
    #pred_token = pred_token * tf.cast(No_padding_token, dtype=tf.float32)
    #print (pred_token)
    pred_token = tf.cast( tf.math.argmax(pred_token, axis=-1) , dtype=tf.int32 ) * No_padding_token
    #print ("----")
    #print (orign_token)
    #print ("----")
    #print (pred_token)
    test_accuracy_token(orign_token,pred_token)
    #tf.print("test_label_acc: ",test_accuracy_label.result(), "test_token_acc: ",test_accuracy_token.result() ,output_stream = filename_test_acc)
    ###
    No_padding_coord = tf.cast(tf.tile(tf.expand_dims(No_padding_token,axis = -1),multiples=[1,1,3]), dtype=tf.float32)
    pred_coord = pred_coord * No_padding_coord
    ###
    No_padding_dist  = tf.tile(tf.expand_dims(pred_token,axis = -1),multiples=[1,1,pred_token.shape[-1]])
    No_padding_dist  = tf.transpose(No_padding_dist, perm=[0,2,1]) +  No_padding_dist 
    #No_padding_dist  = tf.linalg.set_diag(No_padding_dist, tf.zeros([No_padding_dist.shape[0],No_padding_dist.shape[1]],dtype=tf.float32))
    pred_dist = pred_dist * tf.cast( No_padding_dist , dtype=tf.float32 )
    ###
    pathname = './log_COORD_1.00_vDAMN_ln_liq20/'
    filename_label_o = '.o.label.log'
    filename_label_o = pathname + str(epoch + 1) + filename_label_o
    flo = open(filename_label_o,'a')
    filename_label_p = '.p.label.log'
    filename_label_p = pathname + str(epoch + 1) + filename_label_p
    flp = open(filename_label_p,'a')
    tf.print( tf.expand_dims(label,-1), summarize=5000 , output_stream = 'file://'+flo.name )
    tf.print( tf.expand_dims( tf.math.argmax(pred_crystal, axis=-1) ,-1), summarize=5000 , output_stream = 'file://'+flp.name )
    ###label
    filename_coord_o = '.o.coord.log'
    #filename_dist_o = '.o.dist.log'
    #filename_token_o = '.o.token.log'
    filename_coord_o = pathname + str(epoch + 1) + filename_coord_o
    #print(filename_coord_o)
    fco = open(filename_coord_o,'w')
    #filename_dist_o = pathname + str(epoch + 1) + filename_dist_o
    #fdo = open(filename_dist_o,'a')
    #filename_token_o = pathname + str(epoch + 1) + filename_token_o
    #fto = open(filename_token_o,'w')
    filename_coord_m = '.m.coord.log'
    #filename_dist_m = '.m.dist.log'
    filename_token_m = '.m.token.log'
    filename_coord_m = pathname + str(epoch + 1) + filename_coord_m
    fcm = open(filename_coord_m,'w')
    #filename_dist_m = pathname + str(epoch + 1) + filename_dist_m
    #fdm = open(filename_dist_m, 'a')
    #filename_token_m = pathname + str(epoch + 1) + filename_token_m
    #ftm = open(filename_token_m, 'w')
    filename_coord_p = '.p.coord.log'
    #filename_dist_p = '.p.dist.log'
    #filename_token_p = '.p.token.log'
    filename_coord_p = pathname + str(epoch + 1) + filename_coord_p
    fcp = open(filename_coord_p,'w')
    #filename_dist_p = pathname + str(epoch + 1) + filename_dist_p
    #fdp = open(filename_dist_p, 'a')
    #filename_token_p = pathname + str(epoch + 1) + filename_token_p
    #ftp = open(filename_token_p,'w')
    ##///
    tf.print( coord ,summarize=5000, output_stream = 'file://'+fco.name)
    tf.print( mask_coord,summarize=5000, output_stream = 'file://'+fcm.name )
    tf.print( pred_coord, summarize=5000 , output_stream = 'file://'+fcp.name )
    ##///
    #tf.print( tf.expand_dims(orign_token, axis=-1),summarize=121, output_stream = 'file://'+ fto.name )
    #tf.print( tf.expand_dims(mask_token, axis=-1),summarize=121 , output_stream = 'file://'+ ftm.name )
    #tf.print( tf.expand_dims(pred_token, axis=-1),summarize=121, output_stream = 'file://'+ftp.name )
    #tf.print( tf.reshape(dist, [-1,1]),summarize=121, output_stream = 'file://'+fdo.name )
    #tf.print( tf.reshape(pred_dist, [-1,1]),summarize=121, output_stream = 'file://'+fdp.name )
    fco.close()
    fcm.close()
    fcp.close()
    #fto.close()
    #ftm.close()
    #ftp.close()
    flo.close()
    flp.close()



import time

filename_test_acc =open( './log_COORD_1.00_vDAMN_ln_liq20/test_dataset_acc.log','a')
filename_train_acc =open( './log_COORD_1.00_vDAMN_ln_liq20/train_dataset_acc.log','a')
test_accuracy_label = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_label_accuracy')
test_accuracy_token = tf.keras.metrics.CategoricalAccuracy(
        name='test_token_accuracy')
test_total_loss = tf.keras.metrics.Mean(name='test_total_loss')
test_token_loss = tf.keras.metrics.Mean(name='test_token_loss')
test_crystal_loss = tf.keras.metrics.Mean(name='test_crystal_loss')
test_dist_loss = tf.keras.metrics.Mean(name='test_dist_loss')
test_coord_loss = tf.keras.metrics.Mean(name='test_coord_loss')
###############################################################################################
##########     freeze layer   #################################################################
###############################################################################################
model.embed_tokens.trainable=True          #1
model.encoder.trainable=True               #2
model.lm_head.trainable=True               #3
model.gbf_proj.trainable=True              #4
model.gbf.trainable=True                   #5
model.pair2coord_proj.trainable=True       #6
model.dist_head.trainable=True             #7
model.classification_heads.trainable=True  #8 
##############################################################################################
##########    frezze finish  #################################################################
##############################################################################################
##########   test ####
tf.print("embed_tokens.trainable: ",model.embed_tokens.trainable)
tf.print("encoder.trainable: ",model.encoder.trainable)
tf.print("lm_head.trainable: ",model.lm_head.trainable)
tf.print("gbf_proj.trainable: ", model.gbf_proj.trainable)
tf.print("gbf.trainable: ",model.gbf.trainable)
tf.print("pair2coord_proj.trainable: ",model.pair2coord_proj.trainable)
tf.print("dist_head.trainable: ",model.dist_head.trainable)
tf.print("classification_heads.trainable: ",model.classification_heads.trainable)
######################
#update_var_list = []
#tvars = model.trainable_variables
#tf.print("layer_classification_heads: ", model.classification_heads.trainable)
#    tf.print("tvar_name:", layer)
#for layer in model.layers:
#    tf.print(layer)

for epoch in range(1000000):
    start = time.time()
    train_total_loss.reset_states()
    train_token_loss.reset_states()
    train_crystal_loss.reset_states()
    train_dist_loss.reset_states()
    train_coord_loss.reset_states()
    train_accuracy_label.reset_states()
    #train_accuracy_token.reset_states()
    #
    test_total_loss.reset_states()
    test_token_loss.reset_states()
    test_crystal_loss.reset_states()
    test_dist_loss.reset_states()
    test_coord_loss.reset_states()
    test_accuracy_label.reset_states()
    test_accuracy_token.reset_states()
    #
    #Boundary = bool(random.choice([True,False]))
    #TRain_Bo = bool(1-Boundary)
    cut_off = random.uniform(7.5,8)
    #cut_off = 8
    data_Feeder = Data_Feeder(
        path_coord, max_batch_system = 2, max_atoms = 130,
        len_dictionary = len(dictionary), d_cut = cut_off
        ,Training = True
    )
    batch_dataset = data_Feeder.Generate_batch(batch_size=200,Repeat_size=1, shuffle_size=10000)
    for (batch, (o_token, label, o_edge, o_coord, o_dist)) in enumerate(batch_dataset):
        #tf.print("embed_tokens.trainable: ",model.embed_tokens.trainable)
        #tf.print("encoder.trainable: ",model.encoder.trainable)
        #tf.print("lm_head.trainable: ",model.lm_head.trainable)
        #tf.print("gbf_proj.trainable: ", model.gbf_proj.trainable)
        #tf.print("gbf.trainable: ",model.gbf.trainable)
        #tf.print("pair2coord_proj.trainable: ",model.pair2coord_proj.trainable)
        #tf.print("dist_head.trainable: ",model.dist_head.trainable)
        #tf.print("classification_heads.trainable: ",model.classification_heads.trainable)
        train_step(o_token, label, o_edge, o_coord, o_dist, len(dictionary))
        #print ('================')
        #print (train_loss.result())
        #print (train_accuracy.result())
        #print ('================')
        if batch % 100 == 0:
            #print ('Epoch {} Batch {} total_Loss {:.4f} '.format(
            #    epoch + 1, batch, train_total_loss.result()))
            #print ('Epoch {} Batch {} Accuracy_crystal {:.4f}'.format(
            #    epoch + 1, batch, train_accuracy_label.result()))
            tf.print("Epoch:", epoch + 1, " batch: ", batch, "lr: ", optimizer.learning_rate.numpy(),
                    "cut_off:",cut_off," Total loss: ", train_total_loss.result(),
                    " token_loss: ",train_token_loss.result(), "crystal_loss: ", train_crystal_loss.result(),
                    " dist_loss: ", train_dist_loss.result(), "coord_loss:", train_coord_loss.result(),
                    " train_accuracy_label: ", train_accuracy_label.result(),
                    output_stream='file://'+filename_train_acc.name)
            #tf.compat.v1.logging.info('Epoch {} Batch {} total_Loss {:.4f} Accuracy_crystal {:.4f}'.format(
            #    epoch + 1, batch, train_total_loss.result(), train_accuracy_label.result()))
            #tf.compat.v1.logging.flush()
        if batch % 2000 == 0:
            ckpt_save_path = ckpt_manager.save()
        #
    if (epoch + 1) % 1 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))

        test_data_Freeder = Data_Feeder(
                path_test, max_batch_system = 80 , max_atoms = 130,
                len_dictionary = len(dictionary), d_cut = 7# cut_off
                #
                , Training=False
                # learn lable should be False; learn coord shouble be True
        )
        test_data = test_data_Freeder.Generate_test_batch(batch_size=80)
        noise_size = random.uniform(0.0,0.0)
        for (batch, (o_token, label, o_edge, o_coord, o_dist)) in enumerate(test_data):
        	test_step(o_token, label, o_edge, o_coord, o_dist, len(dictionary), epoch, noise_size)
        tf.print("EPOCH: ",epoch + 1,"cut_off:",cut_off," test_label_acc: ",test_accuracy_label.result(), "test_token_acc: ",test_accuracy_token.result() ,
                "test_total_loss: ",test_total_loss.result(),"test_token_loss: ",test_token_loss.result(),
                "test_crystal_loss: ",test_crystal_loss.result(),"test_dist_loss: ",test_dist_loss.result(), 
                "test_coord_loss: ",test_coord_loss.result(), "noise_set: ", noise_size,
                output_stream = 'file://'+filename_test_acc.name)
    print ('Epoch {} total_Loss {:.4f} Accuracy_crystal {:.4f}'.format(epoch + 1,
                                                                       train_total_loss.result(), 
                                                                       train_accuracy_label.result()))
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    tf.keras.backend.clear_session()

filename_test_acc.close()
