#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --job-name=hpc
#SBATCH --mail-type=END
#SBATCH --mail-user=ys3202@nyu.edu
#SBATCH --output=slurm_%j.out
  
module purge
#module load tensorflow/python3.5/1.4.0 
#imodule load cudnn/8.0v6.0
#module load cuda/8.0.44
#RUNDIR=$home/ys3202/dark/run-${SLURM_JOB_ID/.*}
#mkdir -p $RUNDIR
#module load python3/intel/3.6.3
#source /home/yw1007/myenv/bin/activate

# model_idx: 
# - 4 One layer convolution + R2Unet
# - 5 R2Unet + R2Unet
# - 7 inception + R2Unet

# - C_model: C model name used as the first-phase model
# - loss_weight: weight for the loss function
# - normalize: should be the same as C model
# - vel: should be the same as C model


./omp-scan
