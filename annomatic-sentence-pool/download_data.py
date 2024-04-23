from dotenv import load_dotenv
import wandb
import os

def download_data(project_name, tag, root):
    # Get the runs
    runs = wandb.Api().runs(project_name, filters={"tags": tag})

    for run in runs:
        # split the model name to get the output name
        parts = run.config['model'].split('/')
        if len(parts) == 2:
            csv_name = parts[1] + ".csv"
        else:
            csv_name = parts[0] + ".csv"

        # download the csv file
        if csv_name in [x.name for x in run.files()]:
            csv_file = [x for x in run.files() if x.name == csv_name][0]
            csv_file.download(root=root, replace=True)

if __name__ == '__main__':
    # Load environment variables from the .env file
    load_dotenv()
    wandb.login(key=os.environ['WANDB'])

    project_name = 'media-bias-group/annomatic_dataset'

    # Download the data
    download_data(project_name, 'MA', 'data/annotation/ma')
    download_data(project_name, 'rest_1', 'data/annotation/rest_1')
    download_data(project_name, 'final_part_1', 'data/annotation/final_part_1')
    download_data(project_name, 'final_part_2', 'data/annotation/final_part_2')
    download_data(project_name, 'final_part_3', 'data/annotation/final_part_3')

    download_data(project_name, 'final_part1_1', 'data/annotation/final_part1/part_1')
    download_data(project_name, 'final_part1_2', 'data/annotation/final_part1/part_2')
    download_data(project_name, 'final_part2_1', 'data/annotation/final_part2/part_1')
    download_data(project_name, 'final_part2_2', 'data/annotation/final_part2/part_2')
    download_data(project_name, 'rest1_1', 'data/annotation/rest_1/part_1')
    download_data(project_name, 'rest1_2', 'data/annotation/rest_1/part_2')