#!/bin/bash
#SBATCH --partition=stingy	#gpu_a100
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=2  #8       # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:1             # Using 1 GPU card
#SBATCH --mem=200GB               # Request 50GB memory
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
#conda activate tf-gpu2
#conda activate tensorflow 
#python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
#python3 -e "import tensorflow as tf; print(tf.__version__)"
#python3 dp_LDL_simple_fhi47_100_linear_test.py #dp_LDL_simple_fhi46_100_linear_test.py # dp_LDL_simple_fhi45_test.py  # dp_LDL_simple6.py
#wfl_v10_test_0.99.py
#module load openmpi/4.1.5_gcc
#module load fftw/3.3.8_openmpi
#module load cuda/12.1.0
#module load cudnn/8.9.7.29_cuda12.0
#conda activate tf-gpu2
#conda activate tensorflow

#source /home/liwenli/scratch/GPUMD/DPMD/deepmd-kit/bin/activate /home/liwenli/scratch/GPUMD/DPMD/deepmd-kit
#export LD_LIBRARY_PATH=/home/liwenli/scratch/GPUMD/DPMD/deepmd-kit/lib:$LD_LIBRARY_PATH

#srun -n 1 hostname
while true; do nvidia-smi|grep "Default" >> gpu_usage.log; sleep 60; done &

/home/liwenli/scratch/GPUMD/GPUMD-4.2/src/gpumd
#export LD_LIBRARY_PATH=/home/liwenli/scratch/GPUMD/DPMD/lib:$LD_LIBRARY_PATH

#python3 test.py

#/home/liwenli/scratch/GPUMD/GPUMD-4.2/src/gpumd
#export OMP_NUM_THREADS=1
#lmp -pk gpu 1 -in in.lammps
#lmp -pk gpu 1 -in in.lammps
#perl Auto_PT.pl
#module load gromacs/2020.4

#gmx_mpi_d grompp -f RUN.mdp -c 5.gro -p topol.top -o test.tpr -maxwarn 2
#mpirun -np 1 gmx_mpi_d mdrun -v -deffnm test -ntomp 1 -ntmpi 1 -pinstride 0 -pinoffset 0  -nb gpu  -pin on -gpu_id 0
