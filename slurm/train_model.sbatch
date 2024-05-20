#!/bin/bash
#SBATCH --job-name=tomas-train
#SBATCH -p kisski
#SBATCH -n 10
#SBATCH -g a100:1
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --time=0-00:20:00
#SBATCH --account=etechnik_gpu
#SBATCH -o anno_logs/%x-%j.out

PROJECT_PATH=/user/spinde/u11191/llm-annotations-emnlp24
CWD=$(pwd)
module load anaconda3
module load 2021a GCC/10.3.0
module load 2021a python/3.10.13
module load cuda

source activate annomatic
cd PROJECT_PATH
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH
export TOKENIZERS_PARALLELISM=true
python evaluation/train_final_babe_classifier.py