#!/bin/bash
#SBATCH --account=hygate
#SBATCH --partition=fatfree
#SBATCH --ntasks-per-node=2 
#SBATCH --cpus-per-task=9
#SBATCH --nodes=1
#SBATCH -o out.%J.log
#SBATCH -e err.%J.log
#SBATCH --time=24:00:00

#export OMPI_MCA_btl_openib_verbose=1
export OMPI_MCA_btl=openib,self
export OMPI_MCA_btl_openib_if_include=mlx5_0
export OMPI_MCA_btl_openib_allow_ib=1
export I_MPI_PIN_CELL=core
export SLURM_CPU_BIND=none
export I_MPI_PIN_DOMAIN=auto

time mpirun python PyISV.py