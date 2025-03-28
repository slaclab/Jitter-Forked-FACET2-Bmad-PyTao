#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=139
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --account=m4272
#SBATCH --constraint=cpu
#SBATCH --array=0-8

mamba init
mamba activate Multifidelity

i=$SLURM_ARRAY_TASK_ID

echo "Running on node: $(hostname)"
echo "Array job index: $i"

python oneBunch_sub_jitter.py $i
