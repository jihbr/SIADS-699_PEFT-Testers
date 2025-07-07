import os
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager

# Run evaluation pipeline
def run_evaluation(
    model_name: str,
    tasks: str,
    batch_size: int,
    output_dir: str,
    use_chat_template: bool,
    tokenizer_name: str = None,
    limit: int = None,
):
    print(f"starting evaluation for model: {model_name}")
    print(f"tasks: {tasks}")
    print(f"output_directory: {output_dir}")

    evaluation_tracker = EvaluationTracker(output_dir=output_dir, save_details=True, push_to_hub=False)

    # evaluation pipeline parameters
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory="./custom-usmle-qa.py",
        max_samples=limit,
    )
        
    model_config = TransformersModelConfig(
        model_name=model_name,
        tokenizer=tokenizer_name if tokenizer_name else model_name,
        dtype="float16",
        use_chat_template=use_chat_template,
        batch_size=batch_size,
    )

    # main pipeline
    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config
    )

    # pipeline
    pipeline.evaluate()
    print("evaluation complete")
    pipeline.save_and_push_results()
    pipeline.show_results()


def main():
    tasks_to_run = [
        # USMLE CF and MCF evals
        "community|usmle-qa-cf|5|0",
        "community|usmle-qa-mcf|5|0",
        ]
    
    gcs_bucket_name = "open-llm-finetuning"

    models_eval = [
        # Pre-finetuned
        {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "use_chat_template": False,
            "batch_size": 6,
        },
        # Fully finetuned
        {
            "model_name": "Sirius27/BeingWell_llama2_7b",
            "use_chat_template": False,
            "batch_size": 6,
        },
        # LoRA finetuned
        {
            "model_name": "jihbr/usmle-llama7b-lora",
            "use_chat_template": False,
            "batch_size": 6,
            "tokenizer_name": "meta-llama/Llama-2-7b-hf",
        },
        # DoRA finetuned
        {
            "model_name": "jihbr/usmle-llama7b-dora",
            "use_chat_template": False,
            "batch_size": 6,
            "tokenizer_name": "meta-llama/Llama-2-7b-hf",
        },
    ]

    output_directory = f"gcs://{gcs_bucket_name}/evaluation_results"

    for task_to_run in tasks_to_run:
        for model_details in models_eval:
            run_evaluation(
                model_name=model_details["model_name"],
                tasks=task_to_run,
                batch_size=model_details["batch_size"],
                output_dir=output_directory,
                use_chat_template=model_details["use_chat_template"],
                tokenizer_name=model_details.get("tokenizer_name")
            )

if __name__ == "__main__":
    main()
