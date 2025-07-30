import gc,tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
from .UOPdata import *
from .UOPloss import *
from .UOPtool import *
from .XYZdata import *
from .encoder import *
from .model import *
from .self_attention import *
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)

max_neighbor = 80
max_iterT = 100
max_noiseS = 0.003
dictionary = {'MASK':0, 'C':1, 'O':2, 'N':3, 'H':4, 'CLAS':5, 'TEMP':6, 'PRESS':7}
crystal = {'Paracetamol_I':0,'Paracetamol_II':1, 'Paracetamol_III':2, 'Urea_I':3, 'Urea_II':4, 'Urea_III':5, 'ice_0':6, 'ice_Ih':7,'ice_Ic':8}

score_model = Gen3Dmol_Classify(
        encoder_layers = 10,
        encoder_embed_dim = 4096,
        encoder_ffn_embed_dim = 4096,
        encoder_attention_heads = 32,
        Natom = max_neighbor+4, ###############
        iterT = max_iterT,
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
        num_classes = len(crystal),
        token_class = len(dictionary),
        dictionary = dictionary
        )

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,d_model, warmup_steps=100000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
    def __call__(self, step):
        arg1 = tf.math.rsqrt(tf.cast(step, dtype=tf.float32))
        arg2 = tf.cast(step,dtype=tf.float32) * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


checkpoint_path = "./checkpoint_UOPv3_test_0.0.01/train"
step = tf.Variable(0, name="step")

#l_r = 0.000000001 # CustomSchedule(512)
#optimizer = tf.keras.optimizers.Adam(learning_rate=l_r, beta_1=0.9, beta_2=0.98,
#                                     epsilon=1e-9)
ckpt = tf.train.Checkpoint(score_model)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

l_r = 0.000056 # CustomSchedule(512)
optimizer = tf.keras.optimizers.Adam(learning_rate=l_r, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
train_total_loss = tf.keras.metrics.Mean(name='total_loss')
train_token_loss = tf.keras.metrics.Mean(name='token_loss')
train_noise_loss = tf.keras.metrics.Mean(name='noise_loss')
train_cryst_loss = tf.keras.metrics.Mean(name='cryst_loss')
train_temp_loss = tf.keras.metrics.Mean(name='temp_loss')
train_pres_loss = tf.keras.metrics.Mean(name='press_loss')
train_normx_loss = tf.keras.metrics.Mean(name='norm_x_loss')
train_normpair_loss = tf.keras.metrics.Mean(name='norm_pair_loss')
train_accur_labl = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_token_labl = tf.keras.metrics.SparseCategoricalAccuracy(name='train_token_accuracy')
Noise_creator = create_masks(max_noiseS,token_noise=0.1, iterT = max_iterT, training=True)
Noise_test_creator = create_masks(max_noiseS,token_noise=0.1, iterT = max_iterT, training=False)
loss_function = loss_1(1,0.8,0,1,1,max_iterT)

@tf.function(
        input_signature=[
            tf.TensorSpec([None], tf.int32),                       # label_o
            tf.TensorSpec([None, max_neighbor+4, 1], tf.int32),    # token_o
            tf.TensorSpec([None, max_neighbor+4, 3], tf.float32),  # coord_o
            tf.TensorSpec([None], tf.float32),                     # temp_o
            tf.TensorSpec([None], tf.float32)                      # press_o
        ],experimental_relax_shapes=True
        )
def train_step(label_o, token_o, coord_o, temp_o,press_o):
    #bsz = token_o.shape[0]
    #Natom_l = token_o.shape[1]
    #token_o = tf.reshape(token_o,[bsz,Natom_l])
    token_o = tf.squeeze(token_o, axis=-1)     # (bsz, Natom_l)
    #tf.print("UOP main-- label_o.shape",label_o.shape,"token_o.shape",token_o.shape,"coord_o.shape",coord_o.shape,"temp.shape",temp_o.shape,"press_o.shape",press_o.shape)
    
    (input_tokens,input_coords,noise,iter_i) = Noise_creator.Create_noise(token_o,coord_o)
    #tf.print("UOP main-- input_tokens.shape",input_tokens.shape, "input_coords.shape",input_coords.shape,"noise.shape",noise.shape,"iter_i.shape",iter_i)
    
    with tf.GradientTape() as tape:
        out = score_model(input_tokens,input_coords,iter_i,press_o,temp_o,training=True)
        (token_p, label_p, temp_p, press_p, noise_p, x_norm, delta_encoder_pair_rep_norm) = out
        #tf.print("UOP main-- token_p.shape",token_p.shape,"label_p.shape",label_p.shape,"temp_p.shape",temp_p.shape,"press_p.shape",press_p.shape,"noise_p.shape",noise_p.shape,"x_norm:",x_norm,"delta_encoder_pair_rep_norm:",delta_encoder_pair_rep_norm)
        loss = loss_function.ca_loss(token_p, token_o, noise_p, coord_o, noise, label_p, label_o, temp_p,temp_o, press_p,press_o, x_norm, delta_encoder_pair_rep_norm,iter_i)
        (loss_t, token_loss, crystal_loss, coord_loss, temp_loss,press_loss,norm_x,norm_pair,crystal_loss) = loss
        gradients = tape.gradient(loss_t, score_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, score_model.trainable_variables))
        train_total_loss(loss_t)
        train_token_loss(token_loss)
        train_noise_loss(coord_loss)
        train_cryst_loss(crystal_loss)
        train_temp_loss(temp_loss)
        train_pres_loss(press_loss)
        train_normx_loss(norm_x)
        train_normpair_loss(norm_pair)
        #tf.print("UOP main-- label_p vs label_o .shape",label_p.shape,label_o.shape)
        train_accur_labl(label_o,label_p)
        #tf.print("UOP main-- token_p vs token_o .shape",token_p.shape,token_o.shape)
        train_token_labl(token_o,token_p)
    #

def main(epochs):
    print("UOP main-- Hello! this is UOP main!")
    dir_prefixes = ["./Data/Paracetamol/", "./Data/Urea/"]
    filename_log = open('./train_v3.log','a')
    for epoch in range(epochs):
        start = time.time()
        train_total_loss.reset_states()
        train_token_loss.reset_states()
        train_noise_loss.reset_states()
        train_cryst_loss.reset_states()
        train_accur_labl.reset_states()
        train_token_labl.reset_states()
        colletor = Data_Feeder(dir_prefixes,cutoff = 8,max_neighbor=max_neighbor,each_system_batch=10)
        ##===========================================##
        files_in_each_dir = colletor.get_files()
        #for idx, file_list in enumerate(files_in_each_dir):
        #    print(f"UOP main-- {idx+1} :（{dir_prefixes[idx]}）files：")
        #    for f in file_list:
        #        print(f"UOP main--  {f}")
        print("UOP main--===================================")
        ##===========================================##
        batch_data = colletor.Generate_batch(batch_size=10,Repeat_size=1,shuffle_size=10000)
        for (batch, (local_label,local_temp,local_press,local_elements,local_coords)) in enumerate(batch_data):
            #tf.print("test")
            train_step(local_label, local_elements, local_coords, local_temp, local_press)
            if batch %50 == 0:
                tf.print("Epoch:",epoch+1,"batch:",batch,"lr:",optimizer.learning_rate.numpy(),
                        "loss:",train_total_loss.result(), "token:",train_token_loss.result(),":", train_token_labl.result()*100,"%",
                        "noise:",train_noise_loss.result(),#"crystal:",train_cryst_loss.result(),":",train_accur_labl.result()*100,"%",
                        "temp:",train_temp_loss.result(),"press:",train_pres_loss.result(),"norm_x:",train_normx_loss.result(),"norm_pair:",train_normpair_loss.result(),
                        "crystal:",train_cryst_loss.result(),train_accur_labl.result(),
                        output_stream='file://'+filename_log.name)
                tf.print("Epoch:",epoch+1,"batch:",batch,"lr:",optimizer.learning_rate.numpy(),
                        "loss:",train_total_loss.result(), "token:",train_token_loss.result(),":", train_token_labl.result()*100,"%",
                        "noise:",train_noise_loss.result(),#"crystal:",train_cryst_loss.result(),":",train_accur_labl.result()*100,"%",
                        "temp:",train_temp_loss.result(),"press:",train_pres_loss.result(),"norm_x:",train_normx_loss.result(),"norm_pair:",train_normpair_loss.result(),
                        "crystal:",train_cryst_loss.result(),train_accur_labl.result())
        tf.keras.backend.clear_session()
        gc.collect()
        #if batch%5 == 0:
        ckpt_save_path = ckpt_manager.save()
        tf.print('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))
        filename_log.flush()
        os.fsync(filename_log.fileno())
#    test_colletor = Data_Feeder(dir_prefixes,cutoff = 8,max_neighbor=max_neighbor,each_system_batch=1)
#    test_batch_data = colletor.Generate_test_batch(batch_size=1)
#    sample_n = 2
#    sample_i = 0
#    for (batch, (local_label,local_temp,local_press,local_elements,local_coords)) in enumerate(test_batch_data):
#        if tf.random.set_seed() < 0.00001:
#            sample_i = sample_i + 1
#            tf.print(local_coords,summarize=500000)
#            tf.pirnt("========T-100=====generating:")
#            (input_tokens,input_coords,noise,iter_i) = Noise_test_creator.Create_noise(local_elements,local_coords)
#            sampling = DiffusionSampler(score_model,local_coords.shape,max_noiseS,iterT=max_iterT)
#            #(input_tokens,input_coords,noise,iter_i) = Noise_creator.Create_noise(local_elements,local_coords)
#            x_t = sampling.sample(input_tokens,input_coords)
#            tf.print(x_t,summarize=500000)
#        if sample_i >= sample_n:
#            break
#       # tf.keras.backend.clear_session()

if __name__ == "__main__":
    main(5)
