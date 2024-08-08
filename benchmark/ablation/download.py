import os

import wandb
from dotenv import load_dotenv
from tqdm import tqdm

if __name__ == "__main__":
    # Load environment variables from the .env file
    load_dotenv()
    wandb.login(key=os.environ["WANDB_KEY"])

    # Specify the project name and tag
    project_name = "anonymous/annomatic_ablation"
    tag = "zero-shot_system_prompt"

    # Get the runs
    runs = wandb.Api().runs(project_name, filters={"tags": "similarity"})

    for run in tqdm(runs):
        if "2-shot" in run.tags:
            root = "./similarity/2_shot/"

        elif "4-shot" in run.tags:
            root = "./similarity/4_shot/"

        elif "8-shot" in run.tags:
            root = "./similarity/8_shot/"
        else:
            root = "./"

        # split the model name to get the output name
        parts = run.config["model"].split("/")
        if len(parts) == 2:
            csv_name = parts[1] + ".csv"
        else:
            csv_name = parts[0] + ".csv"

        # download the csv file
        if csv_name in [x.name for x in run.files()]:
            csv_file = [x for x in run.files() if x.name == csv_name][0]
            csv_file.download(root=root, replace=True)

    runs = wandb.Api().runs(project_name, filters={"tags": "diversity"})

    for run in tqdm(runs):
        if "2-shot" in run.tags:
            root = "./diversity/2_shot/"

        elif "4-shot" in run.tags:
            root = "./diversity/4_shot/"

        elif "8-shot" in run.tags:
            root = "./diversity/8_shot/"
        else:
            root = "./"

        # split the model name to get the output name
        parts = run.config["model"].split("/")
        if len(parts) == 2:
            csv_name = parts[1] + ".csv"
        else:
            csv_name = parts[0] + ".csv"

        # download the csv file
        if csv_name in [x.name for x in run.files()]:
            csv_file = [x for x in run.files() if x.name == csv_name][0]
            csv_file.download(root=root, replace=True)
