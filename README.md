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
### Accessing the Foundation Model
The notebooks for fine tuning and script for evaluation utilize Llama-3-8b, from HuggingFace, as the foundation model. In order to access this model in HuggingFace:

1. **Login to HuggingFace/Create an Account**  
You need to be logged into HuggingFace to access the foundation model.

2. **Apply to Access Llama-3-8b**  
[Click here to apply to access Llama-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B) (turnaround is usually less than a day).

3. **Generate a PAT**  
[Create a new PAT in HuggingFace](https://huggingface.co/settings/tokens). Assign access to the foundation model in the repositories permissions section. Also in the repositories section, assign write access (This will enable you to push your fine tuned models to your HuggingFace account repository).  

4. **Authentication**  
Before you run any notebooks or scripts accessing the foundation model, open the terminal and enter the following command:  

```bash
huggingface-cli login
```  

You will then be prompted for your HuggingFace username and the PAT you generated. After you enter this information you should have access to the foundation model along with write access to your HuggingFace repositories.  

*Note: If you want to push your fine tuned models to HuggingFace after running the training notebooks then make sure to change the USERNAME variable at the end to your HuggingFace username.  



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
│   ├── peft_prefix_letters.ipynb
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

## Repository Content
Descriptions for the repository content listed above.  

**notebooks/eda.ipynb**  
Initial testing of the data loading function and investigating the label distribution.  

**notebooks/peft_3_8b_prefix_tune.ipynb**   
Prefix Tuning on Llama-3-8b with full text preprocessing (answer as the label/completion).  

**notebooks/peft_config_subsets.ipynb**    
LoRA on a 100 row subset of the data.  

**notebooks/peft_dora.ipynb**  
DoRA on Llama-3-8b with full text preprocessing (answer as the label/completion).  

**notebooks/peft_dora_letters.ipynb**    
DoRA on Llama-3-8b with letter preprocessing (answer_idx as the label/completion).  

**notebooks/peft_lora.ipynb**  
LoRA on Llama-3-8b with full text preprocessing (answer as the label/completion).  

**notebooks/peft_lora_letters.ipynb**  
LoRA on Llama-3-8b with letter preprocessing (answer_idx as the label/completion).  

**notebooks/peft_prefix_letters.ipynb**   
Prefix Tuning on Llama-3-8b with letter preprocessing (answer_idx as the label/completion).  

**notebooks/peft_prefix_tune.ipynb**   
Prefix Tuning on a 100 row subset of the data.  

**notebooks/preprocess-usmle.ipynb**  
Test preprocessing functions used in fine tuning notebooks.  

**src/eval/evaluation.py**   
Script used for evaluating fine tuned models.

**src/helper_functions.py**  
Helper functions used in fine tuning notebooks (function to load the data, preprocessing functions for text and letter strategies, etc...).  

## Licenses  
**Data**  
The USMLE-4-Options Dataset is licensed under [Creative Commons Attribution Share Alike 4.0 International](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/cc-by-sa-4.0.md). To view it click the link or see the LICENSE_DATA file.  

**Foundation Model**  
Llama-3-8b is licensed under [META LLAMA 3 COMMUNITY LICENSE AGREEMENT](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/LICENSE). To view it click the link or see the LICENSE_MODEL file.



