#!/bin/bash
#SBATCH --account=hygate
#SBATCH --partition=premium
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH -o out.%J.log
#SBATCH -e err.%J.log
#SBATCH --time=15:00:00

export CUDA_VISIBLE_DEVICES=""

#export OMPI_MCA_btl_openib_verbose=1
export OMPI_MCA_btl=openib,self
export OMPI_MCA_btl_openib_if_include=mlx5_0
export OMPI_MCA_btl_openib_allow_ib=1
export I_MPI_PIN_CELL=core
export SLURM_CPU_BIND=none
export I_MPI_PIN_DOMAIN=auto

time mpirun python run_training_classification.py
