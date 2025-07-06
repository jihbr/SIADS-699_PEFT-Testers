from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics

# Custom USMLE evaluation

LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}

def prompt_fn(line: dict, task_name: str = None) -> Doc:
    choices = [line["options"][key] for key in sorted(line["options"].keys())]

    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=choices,
        gold_index=LETTER_TO_INDEX[line["answer_idx"]],
        instruction="",
    )

custom_usmle_task = LightevalTaskConfig(
    name="usmle-qa",
    prompt_function=prompt_fn,
    suite=["community"],
    hf_repo="GBaker/MedQA-USMLE-4-options",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    metric=[Metrics.loglikelihood_acc],
    stop_sequence=None,
)

TASKS_TABLE = [custom_usmle_task]

