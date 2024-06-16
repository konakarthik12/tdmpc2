#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --output=slurm_logs/%x/%j.out.log                          # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=slurm_logs/%x/%j.err.log                        # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=12:00:00                                         # how long you would like your job to run; format=hh:mm:ss
#SBATCH --qos=medium                                           # set QOS, this will determine what resources can be requested
#SBATCH --nodes=1                                               # number of nodes to allocate for your job
#SBATCH --cpus-per-task=4                                              # request 4 cpu cores be reserved for your node total
#SBATCH --mem=64gb                                              # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --account=nexus
#SBATCH --partition=tron
#srun -N 1 --mem=512mb bash -c "hostname; python3 --version" &   # use srun to invoke commands within your job; using an '&'
#srun -N 1 --mem=512mb bash -c "hostname; python3 --version" &   # will background the process allowing them to run concurrently
#wait                                                            # wait for any background processes to complete

if [ -n "${SLURM_JOB_ID:-}" ] ; then
    OG_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
    OG_PATH=$(realpath "$0")
fi
cd "$(dirname "$OG_PATH")" || (echo "Could not cd to og script" && exit 1)


fish $OG_PATH/sbatch_cluster.fish "$@"

echo "Running task for real..."
echo "Working directory: $(pwd)"

mamba run --live-stream -n tdmpc2 python ../tdmpc2/tdmpc2/train.py $argv

echo "Task complete"