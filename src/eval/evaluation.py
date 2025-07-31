import torch
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.models.transformers.adapter_model import AdapterModelConfig
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.model_input import GenerationParameters

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


# Run evaluation pipeline

def run_transformer_evaluation(
    model_name: str,
    tasks: str,
    batch_size: int,
    output_dir: str,
    use_chat_template: bool,
    tokenizer_name: str = None,
    limit: int = None,
):
    print(f"Starting transformer evaluation for model: {model_name}")
    print(f"Tasks: {tasks}")
    print(f"Output Directory: {output_dir}")

    evaluation_tracker = EvaluationTracker(output_dir=output_dir, save_details=True, push_to_hub=False)

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory="src/eval/custom_usmle_qa.py",
        max_samples=limit,
    )

    model_config = TransformersModelConfig(
        model_name=model_name,
        tokenizer=tokenizer_name if tokenizer_name else model_name,
        dtype="float16",
        use_chat_template=use_chat_template,
        batch_size=batch_size,
        generation_parameters=GenerationParameters(
            temperature=0.01
        ),
    )

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config
    )

    pipeline.evaluate()
    print(f"Evaluation complete for {model_name}")
    pipeline.save_and_push_results()
    pipeline.show_results()

    # Clean up resources
    del pipeline
    del model_config
    torch.cuda.empty_cache()


def run_adapter_evaluation(
    adapter_weights: bool,
    model_name: str,
    base_model: str,
    tasks: str,
    batch_size: int,
    output_dir: str,
    use_chat_template: bool,
    tokenizer_name: str = None,
    limit: int = None,
):
    print(f"Starting adapter evaluation for adapter: {model_name}")
    print(f"On base model: {base_model}")
    print(f"Tasks: {tasks}")
    print(f"Output Directory: {output_dir}")

    evaluation_tracker = EvaluationTracker(output_dir=output_dir, save_details=True, push_to_hub=False)

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory="src/eval/custom_usmle_qa.py",
        max_samples=limit,
    )
    
    model_config = AdapterModelConfig(
        adapter_weights=adapter_weights,
        model_name=model_name,
        base_model=base_model,
        tokenizer=tokenizer_name if tokenizer_name else base_model,
        dtype="float16",
        use_chat_template=use_chat_template,
        batch_size=batch_size,
        generation_parameters=GenerationParameters(
            temperature=0.01
        ),
    )

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config
    )

    pipeline.evaluate()
    print(f"Evaluation complete for {model_name}")
    pipeline.save_and_push_results()
    pipeline.show_results()

    # Clean up resources
    del pipeline
    del model_config
    torch.cuda.empty_cache()


def main():
    tasks_to_run = [
        # Medical evals
        # "community|usmle-qa-letter|1|0",
        "community|usmle-qa-text|1|0",
        "community|usmle-qa-letter-text|1|0",
        # "community|usmle-qa-mcf|4|0",
        # "community|usmle-qa-cf|4|0",
        # "helm|med_qa|0|0"
        # General evals
        # MMLU
        # HumanEval
        # GSM8K
        # IFEval
        # Hallucination evals
        # ThuthfulQA
        # Safety evals
        # XSTest
        # HarmBench
    ]
    
    gcs_bucket_name = "open-llm-finetuning"
    output_directory = f"gcs://{gcs_bucket_name}/evaluation_results"
    transformer_models_to_eval = [
        # {
        #     "model_name": "meta-llama/Llama-2-7b-hf",
        #     "use_chat_template": False,
        #     "batch_size": 4,
        # },
        # {
        #     "model_name": "meta-llama/Meta-Llama-3-8B",
        #     "use_chat_template": False,
        #     "batch_size": 1,
        # },
        # {
        #     "model_name": "Sirius27/BeingWell_llama2_7b",
        #     "use_chat_template": False,
        #     "batch_size": 4,
        # },
        # {
        #     "model_name": "johnsnowlabs/JSL-MedLlama-3-8B-v1.0",
        #     "use_chat_template": False,
        #     "batch_size": 2,
        # },
    ]

    # Define adapter-based (fine-tuned) models to evaluate
    adapter_models_to_eval = [
        {
            "model_name": "pippalap/llama3-8b-usmle-prefix-letters",
            "base_model": "meta-llama/Meta-Llama-3-8B",
            "adapter_weights": True,
            "use_chat_template": False,
            "batch_size": 1,
            "tokenizer_name": "meta-llama/Meta-Llama-3-8B",
        },
        # {
        #     "model_name": "jihbr/usmle-llama8b-dora-letters-v1",
        #     "adapter_weights": True,
        #     "base_model": "meta-llama/Meta-Llama-3-8B",
        #     "use_chat_template": False,
        #     "batch_size": 1,
        #     "tokenizer_name": "meta-llama/Meta-Llama-3-8B",
        # },
        # {
        #     "model_name": "jihbr/usmle-llama8b-qlora_letters",
        #     "adapter_weights": True,
        #     "base_model": "meta-llama/Meta-Llama-3-8B",
        #     "use_chat_template": False,
        #     "batch_size": 1,
        #     "tokenizer_name": "meta-llama/Meta-Llama-3-8B",
        # },
    ]

    # Loop through each task and run evaluations
    for task in tasks_to_run:

        # # finetuned model
        for model_details in adapter_models_to_eval:
            run_adapter_evaluation(
                adapter_weights=model_details["adapter_weights"],
                model_name=model_details["model_name"],
                base_model=model_details["base_model"],
                tasks=task,
                batch_size=model_details["batch_size"],
                output_dir=output_directory,
                use_chat_template=model_details["use_chat_template"],
                tokenizer_name=model_details.get("tokenizer_name")
            )
        
        # transformer model
        # for model_details in transformer_models_to_eval:
        #     run_transformer_evaluation(
        #         model_name=model_details["model_name"],
        #         tasks=task,
        #         batch_size=model_details["batch_size"],
        #         output_dir=output_directory,
        #         use_chat_template=model_details["use_chat_template"],
        #         tokenizer_name=model_details.get("tokenizer_name")
        #     )



if __name__ == "__main__":
    main()