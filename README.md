# The Promises and Pitfalls of LLM Annotations: A Case Study on Media Bias Detection
This repository hosts the research project "The Promises and Pitfalls of LLM Annotations," which focuses on a case study regarding media bias detection using Large Language Models (LLMs). The project explores how LLMs can be utilized to detect bias in media content and the inherent challenges and opportunities in their annotations.
---
Links to all other published resources: 
- The **Annomatic** python package: [https://github.com/Media-Bias-Group/annomatic](https://github.com/Media-Bias-Group/annomatic)
- The large-scale lexical bias dataset **anno-lexical**: [https://huggingface.co/datasets/mediabiasgroup/anno-lexical](https://huggingface.co/datasets/mediabiasgroup/anno-lexical)
- The **SA-FT** classifier trained on synthetic bias annotations: [https://huggingface.co/mediabiasgroup/roberta-anno-lexical-ft](https://huggingface.co/mediabiasgroup/roberta-anno-lexical-ft)
- The **HA-FT** classifier trained on human bias annotations: [https://huggingface.co/mediabiasgroup/roberta-babe-ft](https://huggingface.co/mediabiasgroup/roberta-babe-ft)



---
## Project Structure
The repository is divided into four main directories, each serving a distinct role in the research project:

### 1. annomatic-benchmark
This directory contains the code necessary to reproduce the benchmark results of the LLM used in our study. It includes scripts for setting up and running the benchmarks, as well as detailed instructions on how to interpret the results.

### 2. annomatic-dataset
Here lies the infrastructure for creating the final dataset used for media bias detection. This includes scripts for dataset assembly, preprocessing, and postprocessing steps necessary to prepare the data for analysis.

### 3. annomatic-sentence-pool
This directory is responsible for the initial phases of data handling, including scraping, cleaning, and transforming data for annotation. It contains tools and scripts that facilitate the extraction and refinement of raw data into a structured format suitable for further processing.

### 4. babe-reannotation
A specialized directory for reannotating a subset of the BABE dataset. It focuses on revising and enhancing the annotations to improve the accuracy and reliability of bias detection in the dataset.

## Getting Started
To get started with this project, clone the repository to your local machine using the following command:

```
git clone https://github.com/yourusername/llm-annotations-media-bias.git
```
#### Prerequisites
Ensure you have the following installed:

- Python 3.8 or higher
- Relevant Python packages as specified in the requirements.txt file

#### Installation
Install the required Python packages by navigating to the repository's root directory and running:

```pip install -r requirements.txt```

#### Usage
Each directory contains its own set of instructions in the form of a README.md file, detailing how to use the scripts and tools provided. It is recommended to follow these instructions sequentially, starting with annomatic-sentence-pool, followed by annomatic-dataset, and then annomatic-benchmark for benchmarking results.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Authors
- Christoph Mandl
- Tomáš Horych
## Acknowledgements
 todo

For more detailed instructions on how to contribute to the project, set up environments, or run tests, refer to the README files within each directory
