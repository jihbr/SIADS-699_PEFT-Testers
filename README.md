# SIADS-699_PEFT-Testers

## Overview
LLMs are powerful general-purpose tools, but they often hallucinate and can be unreliable in specialized fields like medicine or law. Full fine-tuning helps to alleviate this issue, but the process is computationally expensive and time consuming. Parameter-Efficient Fine-Tuning (PEFT) methods offer a solution. These techniques involve training only a small subset of the model's parameters to align it with specialized knowledge without the prohibitive computational cost. This project was designed to evaluate three popular PEFT methods: Low-Rank Adaptation (LoRA), Weight Decomposed Low-Rank Adaptation (DoRA), and Prefix Tuning, to assess their comparative performance, computational cost, and optimal use cases.

## Setup
This project utilizes Vertex AI Workbench from Google Cloud Platform (GCP). In order to source the necessary compute, [GCP offers a free $300 credit to new customers](https://cloud.google.com/free?_gl=1*8qysm5*_ga*MTYzNzQ4MjUwMy4xNzQ4OTA2NDEz*_ga_WH2QY8WWF5*czE3NTM0Nzg3MzQkbzMyJGcxJHQxNzUzNDc4OTczJGo2MCRsMCRoMA..&hl=en).  

The machine type used is g2-standard-8 (Graphics Optimized: 1 NVIDIA L4 GPU, 8 vCPUs, 32GB RAM), and the GPU instance attached is NVIDIA L4 x 1.

### ***Brandon and Pooja maybe here you could explain the specific steps needed for attaching the GPU

### Cloning the Repository
Once your Vertex AI Workbench environment is setup, the next step is to clone the repository.

First click Open JupyterLab --> Then click the Git tab in the top left of your screen --> Within the Git tab click Clone a Repository --> In the space to enter remote repository url, enter: https://github.com/jihbr/SIADS-699_PEFT-Testers --> Next you will be prompted for your GitHub Username and Personal Access Token (PAT) --> After you enter this information the remote repository will be cloned to your Vertex AI Workbench Environment

## Repository Structure
```
lighteval/
├── notebooks/
│   ├── eda.ipynb
│   ├── peft_3-8b_prefix_tune.ipynb
│   ├── peft_config_subsets.ipynb
│   ├── peft_dora.ipynb
│   ├── peft_dora_letters.ipynb
│   ├── peft_lora.ipynb
│   ├── peft_lora_letters.ipynb
│   ├── peft_prefix_tune.ipynb
│   └── preprocess-usmle.ipynb
├── src/
│   ├── eval/
│   │   └── evaluation.py
│   └── helper_functions.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```
