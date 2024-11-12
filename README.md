# Uni-OP
**A universal Order Parameter Architecture for Crystallization**

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
  Traing dataset (different ice crystals) created by [genice](https://github.com/vitroid/GenIce)
author email: liwenl.sim@gmail.com
