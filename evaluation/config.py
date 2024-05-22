import os

from dotenv import load_dotenv

load_dotenv(dotenv_path="local.env")
# load_dotenv()

WANDB_KEY = os.getenv("WANDB_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
PROJECT_NAME = "training-new-babe"
BATCH_SIZE = 32
RANDOM_SEED = 321