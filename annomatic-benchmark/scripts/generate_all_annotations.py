import os
import subprocess

def run_scripts(start_dir):
    # Walk through all directories and files starting from start_dir
    for root, dirs, files in os.walk(start_dir):
        # Check if both download_data.py and eval.py exist in the current directory
        if "download_data.py" in files and "eval.py" in files:
            # Print the current directory
            print(f"Running scripts in {root}")
            try:
                # Change the working directory to root
                os.chdir(root)
                # Run download_data.py
                print("Running download_data.py")
                subprocess.run(["python", "download_data.py"], check=True)
                # Run eval.py
                print("Running eval.py")
                subprocess.run(["python", "eval.py"], check=True)
            except subprocess.CalledProcessError as e:
                # If there's an error during script execution, print it
                print(f"Error running scripts in {root}: {e}")
            finally:
                # Change back to the original start directory
                os.chdir(start_dir)

run_scripts('.')
