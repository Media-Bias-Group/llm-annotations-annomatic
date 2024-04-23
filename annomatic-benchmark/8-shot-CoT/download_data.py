import os

import wandb
from dotenv import load_dotenv
from tqdm import tqdm

if __name__ == "__main__":
    # Load environment variables from the .env file
    load_dotenv()
    wandb.login(key=os.environ["WANDB"], timeout=60)

    # Specify the project name and tag
    project_name = "media-bias-group/annomatic_benchmark"
    tag = "8-shot CoT"

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

    # download the pool
    # pool_path = "media-bias-group/annomatic_benchmark/6qxjmo5h"
    # run = wandb.Api().run(pool_path)

    # csv_name_raw_pool = "final_pool_with_explanation_processed.csv"
    # pool_raw = [x for x in run.files() if x.name == csv_name_raw_pool][0]
    # pool_raw.download(root='./data/', replace=True)
