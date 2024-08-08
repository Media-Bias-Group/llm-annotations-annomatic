import os

from dotenv import load_dotenv

load_dotenv(dotenv_path="local.env")
load_dotenv()

WANDB_KEY=''
HF_TOKEN=''
PROJECT_NAME =''
BATCH_SIZE = 32
RANDOM_SEED = 321