#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --account=m4272
#SBATCH --constraint=cpu
#SBATCH --array=0-20

mamba init
mamba activate Multifidelity

i=$SLURM_ARRAY_TASK_ID

echo "Running on node: $(hostname)"
echo "Array job index: $i"

python twoBunch_sub_jitter.py $i
