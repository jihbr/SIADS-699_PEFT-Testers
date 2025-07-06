# Evaluation

import os
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager

# Run evaluation

def run_evaluation(
    model_name: str,
    tasks: str,
    batch_size: int,
    output_dir: str,
    use_chat_template: bool,
    limit: int = None,
):
    print(f"starting evaluation for model: {model_name}")
    print(f"tasks: {tasks}")

    print(f"output_directory: {os.path.abspath(output_dir)}")
    os.makedirs(output_dir, exist_ok=True)

    evaluation_tracker = EvaluationTracker(output_dir=output_dir, save_details=True, push_to_hub=False)

    # evaluation pipeline parameters
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory="./custom-usmle-qa.py",
        max_samples=limit,
    )

    model_config = TransformersModelConfig(
        model_name=model_name,
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
    task_to_run = "community|usmle-qa|0|0"
    # task_to_run = "community|usmle-qa|0|0,helm|truthfulqa|0|0"

    models_eval = [
        {
            "model_name": "openai-community/gpt2",
            "use_chat_template": False,
            "batch_size": 1,
            # "limit": 20
        }
    ]

    base_output_dir = "../../outputs/evaluation_results"

    for model_details in models_eval:
        model_output_dir = os.path.join(
            base_output_dir, model_details["model_name"].replace("/", "_")
        )

        run_evaluation(
            model_name=model_details["model_name"],
            tasks=task_to_run,
            batch_size=model_details["batch_size"],
            output_dir=model_output_dir,
            use_chat_template=model_details["use_chat_template"],
        )

if __name__ == "__main__":
    main()
