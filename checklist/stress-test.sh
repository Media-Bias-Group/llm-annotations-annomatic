#!/bin/bash

# Navigating to the script directory
SCRIPT_PATH="src/scripts/"
# Running the INV_media_bias.py script
echo "Running Investigative Media Bias Analysis..."
python ${SCRIPT_PATH}INV_media_bias.py
echo "Investigative Analysis Complete."

# Running the MFT_media_bias.py script
echo "Running Multifactorial Media Bias Test..."
python ${SCRIPT_PATH}/MFT_media_bias.py
echo "Multifactorial Test Complete."

# Running the DIR_media_bias.py script
echo "Running Direct Media Bias Analysis..."
python ${SCRIPT_PATH}DIR_media_bias.py
echo "Direct Analysis Complete."

# Returning to the original directory

echo "All scripts executed successfully."
