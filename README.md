# **Uni-OP**: A universal Order Parameter Architecture for Crystallization
![model](https://github.com/user-attachments/assets/6b09e08e-56d1-45e7-ae39-32fc9fa669fe)

 ### **Using**   
 ```text
├── Uni-OP/
│   ├── POSCAR_npy_displacement.py
│   ├── Uni-OP_v0.2_testing.py
│   ├── cont_test.pl
│   └── cont_test.sh
├── SAVE/
│   ├── 1.0_DAMN_liq20/
│       ├── MultiPT/
│           └── only1/ #(**customizable**)
|               ├── *pl *cpp #(Post-processing script)
|               ├── {1..i}.lammpstrj #(output)
|               └── Un-OP_*.txt #(output)
|       └── checkpoint_COORD_1.00_vDAMN_ln_liq20/train/ #(replaceable)
|           └── CHECKPOINT FILE #(replaceable)
│   └── MultiPT/
|       ├── coord/ #(process documentation)
|       ├── dist/ #(process documentation)
|       ├── {1..i}.gro #(replaceable) for your systems
|       └── {1..i}.POSCAR #(replaceable) for your systems
└── Train/
    ├── Data_111/
    |   ├── displacement/coord/*npy
    |   └── *pl *py #(create training data. save in displacement/coord)
    └── Uni_OP_train_v1.py #(training main)
 ```
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
- More data can be added to train the Uni-OP, making it adaptable to more systems

  ### **Training**
  1. Traing dataset (different ice crystals) created by [genice](https://github.com/vitroid/GenIce)

     in Train/Data_111/create_ice_dis2.pl

     Only standard crystals are required.
     
  3. delete Hydrogen and virtual atoms, and create the POSCAR file

     by using [ovito](https://www.ovito.org/docs/current/python/) (in Train/Data_111/ovitos_gro_poscar_d.py)
     
  4. POSCAR -> coord/*.npy by using [pymatgen](https://pymatgen.org/)
 
     in Train/Data_111/POSCAR_npy_displacement.py

  5. Adding the Train_path in Train/Uni_OP_train_v1.py
     ```python
     path_coord = glob.glob('./Data_111/displacement/coord/*.npy')
                + glob.glob('./Data_111/displacement2/coord/*.npy')
                + glob.glob('./Data_111/displacement3/coord/*.npy')
                + glob.glob('./Data_111/liq/coord/*.npy')
                + glob.glob('./Data_111/displacement3/coord/*.npy')
                + ...
     ```
  6. Carrying out Train/Uni_OP_train_v1.py and Training new models
  ### **Loss Function**
  the number “131” = "130" + "1". "130" represents the maximum number of particles contained in a Local structure. "1" represents the central atom.

  if a local structure have more than 131 particles, need make some changes

  Parts of loss:
  - accuracy of predicated token
  - predicated coord - standard coord
  - predicated dist - standard dist
  - accuracy of predicated classification (also can change this classifer to predicate a certain vale)
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
    ...
  ```
### **References**
- [Do Transformers Really Perform Bad for Graph Representation?](https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html)
- [Uni-Mol: A Universal 3D Molecular Representation Learning Framework](https://chemrxiv.org/engage/chemrxiv/article-details/628e5b4d5d948517f5ce6d72)

author email: liwenl.sim@gmail.com
