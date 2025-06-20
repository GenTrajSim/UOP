import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import self_attention

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
        #
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
