from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics

# Custom USMLE evaluation

LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}

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

def usmle_letter_prompt_fn(line: dict, task_name: str = None) -> Doc:
    main_choices = [f"{key}. {value}" for key, value in sorted(line["options"].items())]
    main_query = (
        "Choose the single best answer for the following multiple-choice question. "
        "Respond with only the capital letter of the correct option.\n\n"
        f"Question: {line['question']}\n" + "\n".join(main_choices) + "\n\n"
        "Correct option letter:"
    )

    correct_letter = line["answer_idx"]

    return Doc(
        task_name=task_name,
        query=main_query,
        choices=[correct_letter],
        gold_index=0,
        instruction="",
    )

def usmle_text_prompt_fn(line: dict, task_name: str = None) -> Doc:
    main_choices = [f"{key}. {value}" for key, value in sorted(line["options"].items())]
    main_query = (
        "The following is a multiple-choice question about medicine. "
        "Please write out the full text of the correct answer.\n\n"
        f"Question: {line['question']}\n" + "\n".join(main_choices) + "\n\n"
        "Correct answer:"
    )

    correct_answer_text = line["options"][line["answer_idx"]]

    return Doc(
        task_name=task_name,
        query=main_query,
        choices=[correct_answer_text],
        gold_index=0,
        instruction="",
    )

def usmle_letter_text_prompt_fn(line: dict, task_name: str = None) -> Doc:
    main_choices = [f"{key}. {value}" for key, value in sorted(line["options"].items())]
    main_query = (
        "Choose the best answer and respond with the letter and answer text exactly as shown.\n\n"
        f"Question: {line['question']}\n" + "\n".join(main_choices) + "\n\n"
        "Correct option:"
    )

    correct_combined = f"{line['answer_idx']}. {line['options'][line['answer_idx']]}"

    return Doc(
        task_name=task_name,
        query=main_query,
        choices=[correct_combined],
        gold_index=0,
        instruction="",
    )

task_letter = LightevalTaskConfig(
    name="usmle-qa-letter",
    prompt_function=usmle_letter_prompt_fn,
    suite=["community"],
    hf_repo="GBaker/MedQA-USMLE-4-options",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="sequential",
    metrics=[Metrics.quasi_exact_match],
    generation_size=100,
    stop_sequence=["\n", ".", ","]
)

task_text = LightevalTaskConfig(
    name="usmle-qa-text",
    prompt_function=usmle_text_prompt_fn,
    suite=["community"],
    hf_repo="GBaker/MedQA-USMLE-4-options",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="sequential",
    metrics=[Metrics.quasi_exact_match],
    generation_size=100,
    stop_sequence=["\n", ".", ","]
)

task_letter_text = LightevalTaskConfig(
    name="usmle-qa-letter-text",
    prompt_function=usmle_letter_text_prompt_fn,
    suite=["community"],
    hf_repo="GBaker/MedQA-USMLE-4-options",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="sequential",
    metrics=[Metrics.quasi_exact_match],
    generation_size=100,
    stop_sequence=["\n", ".", ","]
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
    metrics=[Metrics.loglikelihood_acc],
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
    metrics=[Metrics.loglikelihood_acc_norm],
)

TASKS_TABLE = [task_mcf, task_cf, task_letter, task_text, task_letter_text]