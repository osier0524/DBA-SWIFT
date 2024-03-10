#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=SWIFT     # sets the job name if not set from environment
#SBATCH --time=04:00:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger  # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:2
#SBATCH --ntasks=8
#SBATCH --mem 64gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 8 -oversubscribe --allow-run-as-root python $swift_path --config $yaml_path --randomSeed 3782 
mpirun -np 8 -oversubscribe --allow-run-as-root python $swift_path --config $yaml_path --randomSeed 24
mpirun -np 8 -oversubscribe --allow-run-as-root python $swift_path --config $yaml_path --randomSeed 332
mpirun -np 8 -oversubscribe --allow-run-as-root python $swift_path --config $yaml_path --randomSeed 1221 
mpirun -np 8 -oversubscribe --allow-run-as-root python $swift_path --config $yaml_path --randomSeed 1331 
