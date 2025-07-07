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

def prompt_fn_mcf(line: dict, task_name: str = None) -> Doc:
    choices = [f" {key}. {value}" for key, value in sorted(line["options"].items())]
    query = f"Question: {line['question']}\n" + "\n".join(choices) + "\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=list(sorted(LETTER_TO_INDEX.keys())),
        gold_index=LETTER_TO_INDEX[line["answer_idx"]],
        instruction="",
    )

def prompt_fn_cf(line: dict, task_name: str = None) -> Doc:
    choices = [line["options"][key] for key in sorted(line["options"].keys())]
    query = f"Question: {line['question']}\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=LETTER_TO_INDEX[line["answer_idx"]],
        instruction="",
    )

task_mcf = LightevalTaskConfig(
    name="usmle-qa-mcf",
    prompt_function=prompt_fn_mcf,
    suite=["community"],
    hf_repo="GBaker/MedQA-USMLE-4-options",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc],
)

task_cf = LightevalTaskConfig(
    name="usmle-qa-cf",
    prompt_function=prompt_fn_cf,
    suite=["community"],
    hf_repo="GBaker/MedQA-USMLE-4-options",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
)

TASKS_TABLE = [task_mcf, task_cf]