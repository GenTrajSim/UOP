# Uni-OP
#A universal Order Parameter Architecture for Crystallization
#

 USING   working sub_pathname: ${ca_filenmae}, dealing filenames: ${1..i}.gro and {1..i}.POSCAR
 
 testing Data in SAVE/MultiPT/${ca_filenmae}/${1..i}.gro  AND  SAVE/MultiPT/${ca_filenmae}/${1..i}.POSCAR
 
--> Uni-OP/cont_test.sh ## submit

--> SAVE/1.0_DAMN_liq20/MultiPT/${ca_filenmae}/{1..i}.lammpstrj ## outputs
