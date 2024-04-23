import os

import wandb
from dotenv import load_dotenv
from tqdm import tqdm

if __name__ == "__main__":
    # Load environment variables from the .env file
    load_dotenv()
    wandb.login(key=os.environ["WANDB"])

    # Specify the project name and tag
    project_name = "media-bias-group/annomatic_benchmark"
    tag = "4-shot"

    # Get the runs
    runs = wandb.Api().runs(project_name, filters={"tags": tag})

    for run in tqdm(runs):
        # split the model name to get the output name
        parts = run.config["model"].split("/")
        if len(parts) == 2:
            csv_name = parts[1] + ".csv"
        else:
            csv_name = parts[0] + ".csv"

        # download the csv file
        if csv_name in [x.name for x in run.files()]:
            csv_file = [x for x in run.files() if x.name == csv_name][0]
            csv_file.download(root="./data/", replace=True)
