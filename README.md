# SIADS-699_PEFT-Testers

## Overview
LLMs are powerful general-purpose tools, but they often hallucinate and can be unreliable in specialized fields like medicine or law. Full fine-tuning helps to alleviate this issue, but the process is computationally expensive and time consuming. Parameter-Efficient Fine-Tuning (PEFT) methods offer a solution. These techniques involve training only a small subset of the model's parameters to align it with specialized knowledge without the prohibitive computational cost. This project was designed to evaluate three popular PEFT methods: Low-Rank Adaptation (LoRA), Weight Decomposed Low-Rank Adaptation (DoRA), and Prefix Tuning, to assess their comparative performance, computational cost, and optimal use cases.

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
