#!/bin/bash
#PBS -l walltime=20:30:0,nodes=1:xe:ppn=32,gres=shifter

cd ${PBS_O_WORKDIR}

module load shifter

shifterimg pull luntlab/cs547_project:latest

#RESUME_ARGS="--run_id 33iy6q8g --resume"
RESUME_ARGS=""

export OMP_NUM_THREADS=32
aprun -b -n 1 -d 32  -- shifter --image=docker:luntlab/cs547_project:latest \
    --module=mpich,gpu -- python src/train_main.py -v \
    ${RESUME_ARGS} \
    --wandb_tags debug,bw --model LowDNewModel --epochs 1000
