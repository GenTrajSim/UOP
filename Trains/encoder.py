import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import self_attention_plus_embeding

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
              embeding_bias,   #shape (bsz, seq_len, seq_len)
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
            embeding_bias,
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
