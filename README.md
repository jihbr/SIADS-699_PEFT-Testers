# SIADS-699_PEFT-Testers

## Overview
LLMs are powerful general-purpose tools, but they often hallucinate and can be unreliable in specialized fields like medicine or law. Full fine-tuning helps to alleviate this issue, but the process is computationally expensive and time consuming. Parameter-Efficient Fine-Tuning (PEFT) methods offer a solution. These techniques involve training only a small subset of the model's parameters to align it with specialized knowledge without the prohibitive computational cost. This project was designed to evaluate three popular PEFT methods: Low-Rank Adaptation (LoRA), Weight Decomposed Low-Rank Adaptation (DoRA), and Prefix Tuning, to assess their comparative performance, computational cost, and optimal use cases.

## Setup
This project utilizes Vertex AI Workbench from Google Cloud Platform (GCP). In order to source the necessary compute, [GCP offers a free $300 credit to new customers](https://cloud.google.com/free?_gl=1*8qysm5*_ga*MTYzNzQ4MjUwMy4xNzQ4OTA2NDEz*_ga_WH2QY8WWF5*czE3NTM0Nzg3MzQkbzMyJGcxJHQxNzUzNDc4OTczJGo2MCRsMCRoMA..&hl=en).  

The machine type used is g2-standard-8 (Graphics Optimized: 1 NVIDIA L4 GPU, 8 vCPUs, 32GB RAM), and the GPU instance attached is NVIDIA L4 x 1.

### ***Brandon and Pooja maybe here you could explain the specific steps needed for attaching the GPU

### Cloning the Repository in Vertex AI Workbench

1. **Open JupyterLab**  
   Click the "Open JupyterLab" option in your Vertex AI Workbench interface.

2. **Access Git Tab**  
   Click the Git tab in the top left navigation panel.

3. **Clone Repository**  
   Within the Git tab, click "Clone a Repository".

4. **Enter Repository URL**  
   In the dialog box, enter: https://github.com/jihbr/SIADS-699_PEFT-Testers

5. **Authentication**  
You'll be prompted for your GitHub username and your [Personal Access Token (PAT)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

7. **Completion**  
After successful authentication, the repository will be cloned to your Vertex AI Workbench environment.

### Installing Dependencies
The project's dependencies are tracked in the requirements.txt, to install them simply paste the following command in your Vertex AI Workbench terminal:

```bash
pip install -r requirements.txt
```
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
