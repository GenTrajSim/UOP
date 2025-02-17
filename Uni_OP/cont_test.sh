#!/bin/bash
#SBATCH --partition=stingy#gpu_a100   #stingy	#gpu_a100
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=1    #8        # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:1             # Using 1 GPU card
#SBATCH --mem=100GB               # Request 50GB memory
#SBATCH --time=0-02:00:00        # Time limit day-hrs:min:sec
#SBATCH --output=22test_gpujob_%j.log   # Standard output
#SBATCH --error=22test_gpujob_%j.err    # Standard error log

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/liwenli/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
   if [ -f "/home/liwenli/miniconda3/etc/profile.d/conda.sh" ]; then
	. "/home/liwenli/miniconda3/etc/profile.d/conda.sh"
   else
        export PATH="/home/liwenli/miniconda3/bin:$PATH"
   fi
fi
unset __conda_setup
# <<< conda initialize <<<

#conda activate tensorflow 
#python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
#python3 -e "import tensorflow as tf; print(tf.__version__)"
#conda deactivate
#	/home/liwenli/scratch/SAVE/MultiPT/P5000_T200
ca_filenmae="only1"
mkdir "../SAVE/1.0_DAMN_liq20/MultiPT/${ca_filenmae}"
mkdir "../SAVE/MultiPT/dist"
mkdir "../SAVE/MultiPT/coord"
#ca_filenmae=$1
## 1..10
for i in {1..1}
do
	filename1="../SAVE/MultiPT/${i}.gro"
	filename2="../SAVE/MultiPT/${i}.POSCAR"
	filename3="../SAVE/MultiPT/56.panding.gro"
	filename4="../SAVE/MultiPT/56.panding.POSCAR"
	filename5="../SAVE/MultiPT/56.panding.npy"
	filename6="../SAVE/MultiPT/dist/56.panding.npy"
	filename66="../SAVE/MultiPT/coord/56.panding.npy"
	filename7="../SAVE/1.0_DAMN_liq20/MultiPT/${ca_filenmae}/panding.lammpstrj"
	filename77="../SAVE/1.0_DAMN_liq20/MultiPT/${ca_filenmae}/lammps_colour.lammpstrj"
	filename8="../SAVE/1.0_DAMN_liq20/MultiPT/${ca_filenmae}/${i}.lammpstrj"
	cp ../SAVE/1.0_DAMN_liq20/MultiPT/program/*pl ../SAVE/1.0_DAMN_liq20/MultiPT/${ca_filenmae}/.
	cp ../SAVE/1.0_DAMN_liq20/MultiPT/program/*cpp ../SAVE/1.0_DAMN_liq20/MultiPT/${ca_filenmae}/.
	g++ ../SAVE/1.0_DAMN_liq20/MultiPT/${ca_filenmae}/cj1.cpp -o ../SAVE/1.0_DAMN_liq20/MultiPT/${ca_filenmae}/cj -lpthread
	conda activate my_pymatgen
	cp "$filename1" "$filename3"
	echo "cp file $filename1 $filename3"
	cp "$filename2" "$filename4"
	echo "cp file $filename2 $filename4"
	python3 POSCAR_npy_displacement.py $ca_filenmae
	conda deactivate
	conda activate tensorflow
	perl cont_test.pl $ca_filenmae
	cp "$filename77" "$filename8"
	rm "$filename3"
	rm "$filename4"
	rm "$filename66"
	rm "$filename6"
	rm "$filename77"
done
#python3 ovitos_gro_poscar_test.py
#python3 POSCAR_npy_displacement.py
#conda deactivate
#conda activate tensorflow

#perl cont_test.pl
#python3 test_liq_vDAMN2.py
#wfl_v10_test_0.99.py
#module load openmpi/4.0.5_gcc
#module load cuda
#module load gromacs/2020.4

#gmx_mpi_d grompp -f RUN.mdp -c 5.gro -p topol.top -o test.tpr -maxwarn 2
#mpirun -np 1 gmx_mpi_d mdrun -v -deffnm test -ntomp 1 -ntmpi 1 -pinstride 0 -pinoffset 0  -nb gpu  -pin on -gpu_id 0
