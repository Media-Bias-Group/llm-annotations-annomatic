#!/bin/bash
#SBATCH --job-name=tomas-train
#SBATCH -p gpu
#SBATCH -n 10
#SBATCH -G a100:4
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --time=0-01:00:00
#SBATCH --account=etechnik_gpu
#SBATCH -o anno_eval/%x-%j.out

MTL_PATH=/beegfs/zhukova/annomatic-dataset
CWD=$(pwd)
module load 2021a Anaconda3/2021.05
module load 2021a GCC/10.3.0
module load 2021a Python/3.9.5
module load 2021a CUDA/11.4.2

source activate annomatic
cd /beegfs/zhukova/annomatic-dataset
export PYTHONPATH=$PYTHONPATH:$MTL_PATH
export TOKENIZERS_PARALLELISM=true
python evaluation/train_final_classifier.py