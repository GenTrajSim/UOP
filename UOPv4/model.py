import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import self_attention as sat
from .encoder import TransformerEncoderWithPair
from .UOPtool import MaskLMHead,NonLinearHead,GaussianLayer,ClassificationHead,TemperatureHead,PressureHead,DistanceHead,Embeding_PT_iter_P0ro1,Pairwise_mlp,Embeding_iter_token

class Gen3Dmol_Classify(tf.keras.layers.Layer):
    def __init__(
        self, 
        encoder_layers = 15,
        encoder_embed_dim = 512,
        encoder_ffn_embed_dim = 2048,
        encoder_attention_heads = 32,
        Natom = 50,
        iterT = 1000,
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
        token_class = 10,
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
        self.Natom = Natom
        self.iterT = iterT
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
        self.token_class = token_class
        self.dictionary = dictionary
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
                output_dim = self.token_class,
                weight=None
            )
        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K+1, self.encoder_attention_heads #K+1 origin
        )
        self.gbf_proj2=NonLinearHead(
                self.encoder_attention_heads,self.encoder_attention_heads
                )
        self.gbf = GaussianLayer(K, n_edge_type)
        self.ebb = Embeding_PT_iter_P0ro1(self.iterT+1, self.Natom, self.encoder_attention_heads) #######################################################
        self.ebb_token = Embeding_iter_token(self.iterT+1, self.Natom, self.encoder_embed_dim)
        if self.masked_coord_loss > 0:
            self.pair2coord_proj  = NonLinearHead(
                self.encoder_attention_heads, 1
            )
        #if self.masked_dist_loss > 0:
            #self.dist_head = DistanceHead(
            #    self.encoder_attention_heads
            #)
        self.classification_heads = ClassificationHead(input_dim = self.encoder_embed_dim,
                                                       inner_dim = self.encoder_embed_dim,
                                                       num_classes = self.num_classes, 
                                                       #activation_fn, 
                                                       pooler_dropout = self.pooler_dropout)
        self.temperatureHead = TemperatureHead(input_dim=self.encoder_embed_dim,inner_dim = self.encoder_embed_dim,out_dim=1,pooler_dropout = self.pooler_dropout)
        self.pressureHead = PressureHead(input_dim=self.encoder_embed_dim,inner_dim = self.encoder_embed_dim,out_dim=1,pooler_dropout = self.pooler_dropout)
        self.update_nosie = Pairwise_mlp(self.encoder_ffn_embed_dim,3)
    # classmmethod
    # def build_model(cls, args, task): 
    # return cls(args, task.dictionary)
    def call(
        self,
        src_tokens,
        #src_distance,
        src_coord,
        #src_edge_type,
        iter_T,
        press,
        temp,
        encoder_masked_tokens=None,   # always None
        Not_only_features=True,
        PT_predict = True,
        training = False
        #classification_head_name=True
    ):
        #tf.print("src_tokens.shape",src_tokens.shape)
        bsz = src_tokens.shape[0]
        Natom_l = src_tokens.shape[1]
        #src_tokens = tf.reshape(src_tokens,[bsz,Natom_l])
        src_diff = tf.expand_dims(src_coord, axis=-2) - tf.expand_dims(src_coord, axis=-3)
        src_distance = tf.norm(src_diff, axis = -1)
        src_edge_type = (tf.reshape(src_tokens,[-1,src_tokens.shape[-1],1])*len(self.dictionary)) + tf.reshape(src_tokens,[-1,1,src_tokens.shape[-1]])
        #tf.print("src_edge_type.shape",src_edge_type.shape)
        #if not classification_head_name:
        #    features_only = True
        padding_mask = tf.equal(src_tokens, self.padding_idx)
        #if not tf.reduce_any(padding_mask):
        #    padding_mask = None
        x = self.embed_tokens(src_tokens)
        ##############
        x = self.ebb_token(iter_T,x,press,temp)
        #x = self.ebb_token(iter_T,x,press,temp)
        #x = x + embedding_x
        embedding_bias = self.ebb(iter_T,press,temp) ######################################################
        #embedding_bias = tf.broadcast_to(embedding_bias, (bsz, Natom_l, Natom_l, embedding_bias.shape[-1]))
        def get_dist_features(dist, et):
            n_node = dist.shape[-1]
            gbf_feature = self.gbf(dist, et)
            #gbf_feature = tf.concat([gbf_feature,embedding_bias], axis=-1)
            #
            gbf_result = self.gbf_proj(gbf_feature)
            #gbf_result = tf.concat([gbf_result,embedding_bias], axis=-1)
            gbf_result = self.gbf_proj2(gbf_result)
            gbf_result = gbf_result + embedding_bias
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
        encoder_pair_rep = tf.where(tf.math.is_finite(encoder_pair_rep), encoder_pair_rep, tf.cast(tf.zeros_like(encoder_pair_rep),dtype=encoder_pair_rep.dtype))
        encoder_distance = None
        encoder_coord = None
        #if not features_only:
        if self.masked_token_loss > 0:
            logits = self.lm_head(encoder_rep, encoder_masked_tokens)
        if self.masked_coord_loss > 0:
            coords_emb = src_coord
            if padding_mask is not None:
                atom_num = tf.reshape((tf.reduce_sum( 1 - tf.cast(padding_mask, x.dtype) , axis=1) -3), ###need check origin: "-1"
                                      (-1, 1, 1, 1))
            else:
                atom_num = tf.shape(src_coord, 1) - 3 ###?????check!!!!!!!!!!!!!!!!!!!!!! #need check origin: (-1)
            delta_pos = tf.expand_dims(coords_emb, axis=1) - tf.expand_dims(coords_emb, axis=2)
            delta_encoder_pair_rep.set_shape([None, None, None, self.encoder_attention_heads])
            attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
            coord_update = tf.cast( delta_pos ,dtype=attn_probs.dtype)/ atom_num * attn_probs
            coord_update = tf.reduce_sum(coord_update, axis=2)
            up_noise = self.update_nosie(coord_update)
            #encoder_coord = coords_emb + coord_update   ###### coord_update noise output
        #if self.masked_dist_loss > 0:
            #encoder_distance = self.dist_head(encoder_pair_rep)
        if Not_only_features:
            logits_h = self.classification_heads(encoder_rep)
        if PT_predict:
            temp = self.temperatureHead(encoder_rep)
            press = self.pressureHead(encoder_rep)
            return (
                logits,   ###token type
                logits_h,  ##crystal type
                temp,
                press,
                #encoder_distance,
                #encoder_coord,
                up_noise, #coord_update,
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

if __name__ == "__main__":
    
    dictionary = {'MASK':0, 'C':1, 'O':2, 'N':3, 'H':4, 'CLAS':5, 'TEMP':6, 'PRESS':7} 
    test_layer = Gen3Dmol_Classify(
            encoder_layers = 3,
            encoder_embed_dim = 512,
            encoder_ffn_embed_dim = 512,
            encoder_attention_heads = 8,
            Natom = 5+2,
            iterT = 1000,
            dropout = 0.1,
            emb_dropout = 0.1,
            attention_dropout = 0.1,
            activation_dropout = 0.0,
            pooler_dropout = 0.0,
            max_seq_len = 5,
            #activation_fn = "rel
            #pooler_activation_fn = "tanh",
            post_ln = False,
            masked_token_loss = 1.0,
            masked_token_pred = 1.0,
            masked_coord_loss = 1.0,
            masked_dist_loss = 1.0,
            x_norm_loss = 1.0,
            delta_pair_repr_norm_loss = 1.0,
            num_classes = 6,
            token_class = len(dictionary),
            dictionary = dictionary
            )
    token = tf.constant([[4,5,6,1,1,2,1],[4,5,6,2,3,0,0]])
    tf.print("token.shape",token.shape)
    tf.print(token)
    #NO_padding_mask = tf.cast(tf.not_equal(token, 0), dtype=tf.int32)
    #NO_padding_clas = tf.cast(tf.not_equal(token, len(dictionary)-1), dtype=tf.int32)
    #NO_padding = NO_padding_mask * NO_padding_clas
    #tf.print(NO_padding)
    #weight_token = tf.cast((tf.random.uniform(shape=NO_padding.shape))*(len(dictionary)-2)+1, dtype=tf.int32) 
    #tf.print(weight_token*NO_padding)
    #tf.print(tf.where(tf.not_equal( weight_token*NO_padding ,0), NO_padding , token))
    edge = (tf.reshape(token,[-1,token.shape[-1],1])*len(dictionary)) + tf.reshape(token,[-1,1,token.shape[-1]])
    tf.print(edge.shape)
    tf.print(edge)
    coord = tf.constant([[[0,0,0],[0,0,0],[0,0,0],[1.2,0.4,3],[3.1,4.3,0.4],[5.0,0.3,1.1],[3,3,0]], [[0,0,0],[0,0,0],[0,0,0],[0.2,0.3,4],[1,1,1],[3,0,1],[0.4,1,4]]])
    tf.print(coord.shape)
    tf.print(coord)
    diff = tf.expand_dims(coord, axis=2) - tf.expand_dims(coord, axis=1)  # [N, N, D]
    dist_matrix = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))  # [N, N]
    tf.print(dist_matrix.shape)
    tf.print(dist_matrix)
    iter_i = tf.random.uniform(shape=(token.shape[0],),minval=0,maxval=1000,dtype=tf.int32)
    PorC = tf.random.uniform(shape=(token.shape[0],),minval=0,maxval=2,dtype=tf.int32)
    out=test_layer(token,coord,iter_i)
    #tf.print(out.shape)
    (
            logits,   ###token type
            logits_h,  ##crystal type
            temp,
            press,
            #encoder_distance,
            #encoder_coord,
            coord_update,
            x_norm,
            delta_encoder_pair_rep_norm
            )=out
    tf.print(logits.shape)
    tf.print(logits_h.shape)
    tf.print(temp.shape)
    tf.print(press.shape)
    #tf.print(encoder_distance.shape)
    #tf.print(encoder_coord.shape)
    tf.print(coord_update.shape)
    tf.print(x_norm.shape)
    tf.print(delta_encoder_pair_rep_norm.shape)
