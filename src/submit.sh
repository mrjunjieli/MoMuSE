#!/bin/bash
#SBATCH -J ImaginNet
#SBATCH -p p-A800
#SBATCH -A t00120220002
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2


source /home/lijunjie/.bashrc
conda activate /mntnfs/lee_data1/lijunjie/anaconda3/envs/py1.11
bash run_tmp.sh