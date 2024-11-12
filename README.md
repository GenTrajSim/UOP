# **Uni-OP**: A universal Order Parameter Architecture for Crystallization
![model](https://github.com/user-attachments/assets/6b09e08e-56d1-45e7-ae39-32fc9fa669fe)

 ### **Using**   
 - working sub_pathname: ${ca_filenmae}, dealing filenames: ${1..i}.gro and {1..i}.POSCAR
 - testing Data in SAVE/MultiPT/${ca_filenmae}/${1..i}.gro  AND  SAVE/MultiPT/${ca_filenmae}/${1..i}.POSCAR
 - --> Uni-OP/cont_test.sh ## **submit** Example:
   ```bash
   ca_filenmae="only1" #working folder only1 -> change for your working folder
   ```
 - --> SAVE/1.0_DAMN_liq20/MultiPT/${ca_filenmae}/{1..i}.lammpstrj ## **outputs**


 ### **Origin model:**
 [Uni-OP_457MB](https://www.dropbox.com/scl/fo/yvcfi23nokcg7u2j37aa0/AMkAqWznc35bRIxMIcHv88c?rlkey=a1isd575voytueqmw0vfttctw&st=94yb40tf&dl=0)
 - Pull this model in SAVE/1.0_DAMN_liq20/checkpoint_COORD_1.00_vDAMN_ln_liq20/train
 - this model only train the 10 kinds of ice crystals and liquid at different P-T conditions
 - this only have 4 kinds of tokens (elements)
   in Uni_OP_train_v1.py and Uni-OP_v0.2_testing.py
   ```python
   dictionary = {'MASK':0, 'C':1, 'O':2, 'CLAS':3}
   #Mask-> adding noises for masking elements;
   #CLAS-> the special token for predicating the classifications of Local structures
   #for training for new elements, need change this part of in Uni_OP_train_v1.py and Uni-OP_v0.2_testing.py
   ```
- dictionary of crystal types (can find in Train/Data_111/create_ice_dis2.pl)
  ```python
  dictionary = {'liquid':0,
                'ice1c':5,
                'ice1h':6,
                'ice2':7,
                'ice3':9,
                'ice4':10,
                'ice5':12,
                'ice6':13,
                'ice7':14,
                'ice0':17,
                'ice12':20}
  ## This model has a total of 31 categories, but only 11 of them are trained and can be supplemented.
  ## If you go beyond 31 categories, you need to further modify the code.
  ```

  ### **Training**
  1. Traing dataset (different ice crystals) created by [genice](https://github.com/vitroid/GenIce)

     in Train/Data_111/create_ice_dis2.pl
     
  2. delete Hydrogen and virtual atoms, and create the POSCAR file

     by using [ovito](https://www.ovito.org/docs/current/python/) (in Train/Data_111/ovitos_gro_poscar_d.py)
     
  3. POSCAR -> coord/*.npy by using [pymatgen](https://pymatgen.org/)
 
     in Train/Data_111/POSCAR_npy_displacement.py

  4. Adding the Train_path in Train/Uni_OP_train_v1.py
     ```python
     path_coord = glob.glob('./Data_111/displacement/coord/*.npy')
                + glob.glob('./Data_111/displacement2/coord/*.npy')
                + glob.glob('./Data_111/displacement3/coord/*.npy')
                + glob.glob('./Data_111/liq/coord/*.npy')
                + glob.glob('./Data_111/displacement3/coord/*.npy')
                + ...
     ```
  5. Carrying out Train/Uni_OP_train_v1.py and Training new models
  ### **Loss Function**
  the number “131” = "130" + "1". "130" represents the maximum number of particles contained in a Local structure. "1" represents the central atom.

  if a local structure have more than 131 particles, need make some changes
  ```python
  def loss_function_1(pred_token, orign_token, 
                    pred_crystal, real_crystal, 
                    new_coord, real_coord, 
                    new_dist, real_dist, 
                    loss_x_norm, loss_delta_encoder_pair_rep_norm, 
                    len_dictionary =4,  ## dictionary = {'MASK':0, 'C':1, 'O':2, 'CLAS':3}
                    masked_token_loss = 1.0, crysral_class_loss =1.0,
                    masked_coord_loss = 1.0, masked_dist_loss =1.0,
                    x_norm_loss =1.0, delta_pair_repr_norm_loss =1.0):
    # because of dictionary = {'MASK':0, 'C':1, 'O':2, 'CLAS':3}, need neglect MASK and CLAS tokens.
    # if adding new new elements, need make some changes
    mask_token = tf.cast(tf.where(tf.not_equal(orign_token, 0)&tf.not_equal(orign_token,3),1,0), dtype=tf.int32) #####   3->classify need change in different object
    pred_t_One = tf.math.argmax(tf.nn.log_softmax(pred_token, axis=-1),-1)
    mask_tokenP= tf.cast(tf.where(tf.not_equal(pred_t_One , 0)&tf.not_equal(pred_t_One ,3),1,0), dtype=tf.int32) #####   3->classify need change in different object
    mask_token_classify = tf.cast(tf.not_equal(orign_token, 0), dtype=tf.int32)
    orign_token = orign_token * mask_token_classify
    token_loss = loss_object(orign_token,tf.nn.log_softmax(pred_token,axis=-1))
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
  ```
### **References**
- [Do Transformers Really Perform Bad for Graph Representation?](https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html)
- [Uni-Mol: A Universal 3D Molecular Representation Learning Framework](https://chemrxiv.org/engage/chemrxiv/article-details/628e5b4d5d948517f5ce6d72)

author email: liwenl.sim@gmail.com
