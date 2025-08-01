import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
        q = q*tf.cast(self.scaling ,dtype=q.dtype)
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
