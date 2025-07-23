import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class loss_1:
    def __init__(self, pre_p, cutoff, dl, c0, function):
        self.pre_p = pre_p
        self.cutoff= cutoff
        self.dl = dl
        self.dictionary = {'MASK':0, 'C':1, 'O':2, 'N':3, 'H':4, 'CLAS':5, 'TEMP':6, 'PRESS':7}
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)
        self.loss_object_cry = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if function ==0:
            self._pre_function = lambda x: 1.0/(x + 1e-8)
        if function ==1:
            self._pre_function = lambda x: ((1 - tf.math.erf( self.dl*(x-self.cutoff)))/2) + c0
        if function ==2:
            self._pre_function = tf.identity
    #def _pre_function_0(self):
        #
    #def _pre_function_1(self):
        #
    #def _pre_function_2(self):
        #

    def ca_loss(self, pred_token, origin_token, pred_noise, origin_coord, origin_nosie, pred_cry, origin_cry, pred_temp, origin_temp, pred_press, origin_press,norm_x,norm_pair):
        #mask_token = tf.cast(tf.where( tf.not_equal(orign_token, 0) & tf.not_equal(orign_token,5) & tf.not_equal(orign_token,6) & tf.not_equal(orign_token,7) ,1,0), dtype=tf.int32)
        # token
        token_loss = self.loss_object(origin_token,tf.nn.log_softmax(pred_token,axis=-1))
        # crystal
        crystal_loss=self.loss_object_cry(origin_cry,tf.nn.log_softmax(pred_cry,axis=-1))
        # dist
        dist = None
        dist = tf.norm(origin_coord, axis=2, keepdims=True)
        pre_d = self.pre_p*self._pre_function(dist)
        ####
        #tf.print("UOP loss-- dist:",dist.shape,"pre_d:",pre_d.shape)
        #
        mask_token = tf.cast(tf.where( tf.not_equal(origin_token, 0) & tf.not_equal(origin_token,5) & tf.not_equal(origin_token,6) & tf.not_equal(origin_token,7) ,1,0), dtype=tf.int32)
        ##
        #tf.print("UOP loss-- mask_token:", mask_token.shape, (origin_nosie * pre_d).shape, (pred_noise * pre_d).shape)
        # coord
        labels = tf.reshape( origin_nosie, [origin_nosie.shape[0],-1]) #* pre_d * tf.cast(tf.tile(tf.expand_dims(mask_token,-1), multiples=[1,1,3]), dtype=tf.float32)
        predictions = tf.reshape( pred_noise, [pred_noise.shape[0], -1] ) #* pre_d * tf.cast(tf.tile(tf.expand_dims(mask_token,-1), multiples=[1,1,3]), dtype=tf.float32)
        coord_loss = tf.keras.losses.MSE(labels, predictions)
        #coord_loss = tf.compat.v1.losses.huber_loss(
        #    labels = origin_nosie * pre_d,
        #    predictions = pred_noise * pre_d,
        #    weights = tf.cast(tf.tile(tf.expand_dims(mask_token,-1), multiples=[1,1,3]), dtype=tf.float32),
        #    delta = 0.01
        #    #reduction=tf.keras.losses.Reduction.NONE
        #)
        loss_temp = tf.keras.losses.MSE(origin_temp,pred_temp)
        loss_press= tf.keras.losses.MSE(origin_press,pred_press)
        loss = token_loss + (coord_loss*10) # + (norm_x + norm_pair)*0.01 #+ (loss_temp*0.001) + (loss_press*0.00001) 
        return loss, token_loss, crystal_loss, coord_loss, loss_temp,loss_press,norm_x,norm_pair

if __name__ == "__main__":
    loss_function = loss_1(1,0.8,4,1,1)
    pred_token = tf.random.uniform(shape=(2, 7, 8), minval=0, maxval=1, dtype=tf.float32) #tf.constant([[5,6,7,1,1,1,2],[5,6,7,1,2,3,0]])
    tf.print(pred_token.shape)
    tf.print(pred_token)
    origin_token =tf.constant([[5,6,7,1,2,1,2],[5,6,7,1,2,3,0]])
    pred_coord = tf.constant([[[0,0,0],[0,0,0],[0,0,0],[0.3,0.1,0],[1,2,3],[0.32,3,6],[2,3,10]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0.1,0.23,1],[2,1.2,3],[0,1,21.2]]])
    origin_coord=tf.constant([[[0,0,0],[0,0,0],[0,0,0],[0.,0.,0],[0.1,2.2,3.2],[3.2,1,3],[2,3,10]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0.01,0.3,1.2],[2.1,1.4,3.1],[10,11,1.2]]]) 
    tf.print(origin_coord.shape)
    tf.print(origin_coord)
    pred_cry = tf.random.uniform(shape=(2,18), minval=0, maxval=1, dtype=tf.float32)#tf.constant([0,1]) 
    origin_cry = tf.constant([1,1])
    tf.print(origin_cry.shape)
    tf.print(origin_cry)
    loss_t = loss_function.ca_loss(pred_token, origin_token, pred_coord, origin_coord, pred_cry, origin_cry)
    tf.print(loss_t)
