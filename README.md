# **UOP**: A universal Order Parameter Architecture for Crystallization

## Environment requirements
The following softwares/environments (versions) are required.

The softwares (versions) are necessary in the "tensorflow" conda environment: 
```bash
python == 3.11.0
tensorflow == 2.4.0
numpy == 1.26.3
scipy == 1.12.0
```
The softwares (versions) are necessary in the "my_pymatgen" conda environment: 
```bash
numpy == 1.26.3
pymatgen == 2024.6.10
```
Operating system: Linux
## Installation guide: N/A
The codes only need to satisfy the above environments and maintain the path format in github in order to run. 
## A small dataset is used to demonstrate the code of Training UOP: 
```bash
Train/Data_111/displacement/coord/*npy
```
(different standard cystals)
## The training code of UOP : 
```bash
Train/Uni_OP_train_v1.py
```
Execute the command:
```bash
python Uni_OP_train_v1.py
```
## A configuration for demonstrating the classification ability of UOP:
The configurations that need to be processed are stored in the following path. And a configuration requires two files in different formats (gro and POSCAR).
```bash
SAVE/MultiPT/*.POSCAR
SAVE/MultiPT/*.gro
```
The ".gro" are the configuration files in Gromacs format. 

The ".POSCAR" files with the same name are the same configuration files (.gro) only containing O atoms.

The file names (prefix) are integers from 1 to *n*. i.e., (1.gro 1.POSCAR)configuration-1, (2.gro 2.POSCAR)configuration-2, (3.gro 3.POSCAR)configuration-3 ...
## The code for predicting the types of local structures in the above configurations:
```bash
Uni_OP/Uni-OP_v0.2_testing.py
Uni_OP/cont_test.pl
Uni_OP/cont_test.sh
```
Execute the command:
```bash
sh cont_test.sh
```
The "Uni-OP_v0.2_testing.py" is the main program of the prediction module.

The "cont_test.pl" and "cont_test.sh" are control programs for batch prediction of configurations.

In line 32 of "cont_test.sh", the working path is specified in "only1". You can assign the ca_filenmae variable ($ca_filenmae) to the desired working pathname.

And the OUTPUT files are saved in "./SAVE/1.0_DAMN_liq20/MultiPT/only1/" or "./SAVE/1.0_DAMN_liq20/MultiPT/$ca_filenmae/"

In line 35 of "cont_test.sh", {1..1}, this sets the total number of configurations to be processed. That is, it can be changed to "{1..*n*}".

The system environment requires two conda subenvironments, namely tensorflow and my_pymatgen. 

- tensorflow: tensorflow-2.4.0
- my_pymatgen: pymatgen 2024.6.10
  
If you need to utilize the newly trained model, the checkpoint files should be stored in the following path: ./SAVE/1.0_DAMN_liq20/checkpoint_COORD_1.00_vDAMN_ln_liq20/train
The model trained in this project can be obtained from the following link:  
[Uni-OP_457MB](https://www.dropbox.com/scl/fo/yvcfi23nokcg7u2j37aa0/AMkAqWznc35bRIxMIcHv88c?rlkey=a1isd575voytueqmw0vfttctw&st=94yb40tf&dl=0)

## Expected OUTPUT

The ouput files are stored in "./SAVE/1.0_DAMN_liq20/MultiPT/$ca_filenmae/".

i.e., 1.lammpstrj .. n.lammpstrj are the final input configuration files with UOP-classification information.
```bash
ITEM: ATOMS id type xu yu zu ice17 ice1c ice1h ice2 ice3 ice4 ice5 ice6 ice7 ice20
1 OW1 3.56 0.07 42.34 0 0.94 0 0 0 0 0 0 0 0
```
The "ice17 ice1c ice1h ice2 ice3 ice4 ice5 ice6 ice7 ice20" labels are the corresponding crystal-like degree (CLD) of 10 crystals for each particle.

## NV4090/v100/A100 test 

![modelfig](https://github.com/user-attachments/assets/4754cb09-5e62-43da-96cf-d8a7ee7a2e30)


 ## **Detailed Using Methods**   
 - workflow
 ```text
├── Uni-OP/
│   ├── POSCAR_npy_displacement.py
│   ├── Uni-OP_v0.2_testing.py
│   ├── cont_test.pl
│   └── cont_test.sh
├── SAVE/
│   ├── 1.0_DAMN_liq20/
│       ├── MultiPT/
│           ├── only1/ #(**customizable**)
|               ├── *pl *cpp #(Post-processing script, copy from "../program/")
|               ├── {1..i}.lammpstrj #(output)
|               └── Un-OP_*.txt #(output)
|           └── program/
|               └── *pl *cpp #(Post-processing scripts)
|       └── checkpoint_COORD_1.00_vDAMN_ln_liq20/train/ #(replaceable)
|           └── MODEL CHECKPOINT FILE #(replaceable)
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
     
     Execute the command:
     ```bash
     perl create_ice_dis2.pl
     ```
     
  3. delete Hydrogen and virtual atoms, and create the POSCAR file

     by using [ovito](https://www.ovito.org/docs/current/python/) (in Train/Data_111/ovitos_gro_poscar_d.py)

     Execute the command:
     ```bash
     python ovitos_gro_poscar_d.py
     ```
  5. POSCAR -> coord/*.npy by using [pymatgen](https://pymatgen.org/)
 
     in Train/Data_111/POSCAR_npy_displacement.py
     
     Execute the command:
     ```bash
     python POSCAR_npy_displacement.py
     ```

  7. Adding the Train_path in Train/Uni_OP_train_v1.py
     ```python
     path_coord = glob.glob('./Data_111/displacement/coord/*.npy')
                + glob.glob('./Data_111/displacement2/coord/*.npy')
                + glob.glob('./Data_111/displacement3/coord/*.npy')
                + glob.glob('./Data_111/liq/coord/*.npy')
                + ...
     ```
  8. Carrying out Train/Uni_OP_train_v1.py and Training new models
     Execute the command:
     ```bash
     python Uni_OP_train_v1.py
     ```
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
The attention network with pairs refers to the following articles.
- [Do Transformers Really Perform Bad for Graph Representation?](https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html)
- [Uni-Mol: A Universal 3D Molecular Representation Learning Framework](https://chemrxiv.org/engage/chemrxiv/article-details/628e5b4d5d948517f5ce6d72)

author email: liwenl.sim@gmail.com
