#!/bin/bash
#SBATCH --job-name=coreset_classifiers
#SBATCH --output=testing.out
#SBATCH --error=err_testing.err
#SBATCH -p kisski
#SBATCH -t 02:00:00
#SBATCH -G A100:1
#SBATCH --mem=40GB
#SBATCH -C inet

PROJECT_PATH=/user/spinde/u11191/llm-annotations
CWD=$(pwd)
module load anaconda3
module load cuda
module load python/3.10.13

eval "$(conda shell.bash hook)"
conda activate llmann_env

# python --version
# python -m torch.utils.collect_env
# nvcc -V

cd "$PROJECT_PATH"
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH
export TOKENIZERS_PARALLELISM=true
# export WANDB_DISABLE_SERVICE=true
echo "Running the python script"
python -u coreset/train_coreset_classifier.py