import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
    def __init__(self, 
            input_dim, 
            inner_dim, 
            num_classes, 
            #activation_fn, 
            pooler_dropout):
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

class TemperatureHead(tf.keras.layers.Layer):
    def __init__(self,
            input_dim,
            inner_dim,
            out_dim,
            #activation_fn,
            pooler_dropout):
        super(TemperatureHead, self).__init__()
        self.dense = tf.keras.layers.Dense(inner_dim)
        #self.activation_fn = tf.keras.activations.tanh() #### need check
        self.activation_fn = tf.keras.layers.Activation('gelu')
        self.dropout = tf.keras.layers.Dropout(pooler_dropout)
        self.out_proj = tf.keras.layers.Dense(out_dim)
    def call (self, features):
        x = features[:, 1, :]
        x = self.dropout(x)
        x = self.dense(x)
        #x = tf.keras.activations.tanh(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class PressureHead(tf.keras.layers.Layer):
    def __init__(self,
            input_dim,
            inner_dim,
            out_dim,
            #activation_fn,
            pooler_dropout):
        super(PressureHead, self).__init__()
        self.dense = tf.keras.layers.Dense(inner_dim)
        #self.activation_fn = tf.keras.activations.tanh() #### need check
        self.activation_fn = tf.keras.layers.Activation('gelu')
        self.dropout = tf.keras.layers.Dropout(pooler_dropout)
        self.out_proj = tf.keras.layers.Dense(out_dim)
    def call (self, features):
        x = features[:, 2, :]
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

class Embeding_PT_iter_P0ro1(tf.keras.layers.Layer):
    def __init__(self, Total_T, NATOMS, hidde1=16, hidde2=16, CorP=2):
        super(Embeding_PT_iter_P0ro1, self).__init__()
        self.embed_T = tf.keras.layers.Embedding(Total_T, hidde1)
        self.embed_C = tf.keras.layers.Embedding(CorP, hidde2)
        #self.dense_P = tf.keras.layers.Dense(hiddeP)
        #self.dense_T = tf.keras.layers.Dense(hiddeT)
        self.NATOMS = NATOMS
        self.denseall= tf.keras.layers.Dense(self.NATOMS*self.NATOMS)
        self.activation_fn =  tf.keras.layers.Activation('gelu')
    def call (self, iter_T, Predict01):
        embedingT = self.embed_T(iter_T)
        embedingC = self.embed_C(Predict01)
        #denseP = self.dense_P(press)
        #denseT = self.dense_T(temp)
        embeding = tf.concat([embedingT, embedingC], axis=-1)
        #embeding = tf.concat([embeding, denseP], axis=-1)
        #embeding = tf.concat([embeding, denseT], axis=-1)
        embeding = self.denseall(embeding)
        embeding = self.activation_fn(embeding)
        embeding = tf.reshape(embeding, (-1, self.NATOMS, self.NATOMS, 1))
        return embeding
