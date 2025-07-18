import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .UOPdata import *
from .UOPloss import *
from .UOPtool import *
from .XYZdata import *
from .encoder import *
from .model import *
from .self_attention import *

max_neighbor = 80
dictionary = {'MASK':0, 'C':1, 'O':2, 'N':3, 'H':4, 'CLAS':5, 'TEMP':6, 'PRESS':7}
crystal = {'Paracetamol_I':0,'Paracetamol_II':1, 'Paracetamol_III':2, 'Urea_I':3, 'Urea_II':4, 'Urea_III':5, 'ice_0':6, 'ice_Ih':7,'ice_Ic':8}
score_model = Gen3Dmol_Classify(
        encoder_layers = 3,
        encoder_embed_dim = 512,
        encoder_ffn_embed_dim = 512,
        encoder_attention_heads = 8,
        Natom = max_neighbor+4, ###############
        iterT = 50,
        dropout = 0.1,
        emb_dropout = 0.1,
        attention_dropout = 0.1,
        activation_dropout = 0.1,
        pooler_dropout = 0.1,
        max_seq_len = max_neighbor+4, ########
        post_ln = False,
        masked_token_loss = 1.0,
        masked_token_pred = 1.0,
        masked_coord_loss = 1.0,
        masked_dist_loss = 1.0,
        x_norm_loss = 1.0,
        delta_pair_repr_norm_loss = 1.0,
        num_classes = 6,
        crytal_class = len(dictionary),
        dictionary = dictionary
        )

def main():
    print("Hello! this is UOP main!")
    dir_prefixes = ["./Data/Paracetamol/", "./Data/Urea/"]
    colletor = Data_Feeder(dir_prefixes,cutoff = 8,max_neighbor=max_neighbor,each_system_batch=1)
    ##===========================================##
    files_in_each_dir = colletor.get_files()
    for idx, file_list in enumerate(files_in_each_dir):
        print(f"{idx+1} :（{dir_prefixes[idx]}）files：")
        for f in file_list:
            print(f"  {f}")
    print("===================================")
    ##===========================================##
    batch_data = colletor.Generate_batch(batch_size=10,Repeat_size=1,shuffle_size=100)
    for (batch, (local_label,local_temp,local_press,local_elements,local_coords)) in enumerate(batch_data):
        tf.print("test")

if __name__ == "__main__":
    main()
