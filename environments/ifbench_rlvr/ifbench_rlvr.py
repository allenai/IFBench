import ast
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

import instructions_registry
import verifiers as vf

try:
    from open_instruct.IFEvalG import instructions_registry as rlvr_instructions_registry
except ImportError:
    rlvr_instructions_registry = None


DEFAULT_TRAIN_DATASET = "allenai/IF_multi_constraints_upto5"
DEFAULT_EVAL_DATASET = "allenai/IFBench_test"


def _normalise_kwargs(kwargs: Any) -> dict[str, Any]:
    if kwargs is None:
        return {}
    if not isinstance(kwargs, dict):
        raise TypeError(f"Expected instruction kwargs to be a dict or None, got {type(kwargs)!r}")
    return {key: value for key, value in kwargs.items() if value is not None}


def _flatten_ground_truth(ground_truth: str) -> tuple[list[str], list[dict[str, Any]]]:
    parsed = ast.literal_eval(ground_truth)
    instruction_ids: list[str] = []
    kwargs: list[dict[str, Any]] = []

    for entry in parsed:
        ids = entry.get("instruction_id", [])
        args = entry.get("kwargs", [])
        if isinstance(ids, str):
            ids = [ids]
        if not isinstance(args, list):
            args = [args]
        if len(args) < len(ids):
            args = [*args, *([{}] * (len(ids) - len(args)))]
        for instruction_id, instruction_kwargs in zip(ids, args):
            instruction_ids.append(instruction_id)
            kwargs.append(_normalise_kwargs(instruction_kwargs))

    return instruction_ids, kwargs


def normalise_example(example: dict[str, Any]) -> dict[str, Any]:
    """Convert released IFBench train/eval formats into a verifiers task row."""
    if "prompt" in example:
        prompt = example["prompt"]
        instruction_ids = list(example["instruction_id_list"])
        kwargs = [_normalise_kwargs(item) for item in example["kwargs"]]
    else:
        messages = example["messages"]
        prompt = next(message["content"] for message in messages if message["role"] == "user")
        instruction_ids, kwargs = _flatten_ground_truth(example["ground_truth"])

    answer_payload = {
        "key": example.get("key"),
        "prompt": prompt,
        "instruction_id_list": instruction_ids,
        "kwargs": kwargs,
    }

    return {
        "question": prompt,
        "answer": json.dumps(answer_payload, ensure_ascii=False),
        "info": {
            "key": example.get("key"),
            "instruction_id_list": instruction_ids,
        },
    }


def _read_jsonl(path: str | Path) -> Dataset:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(normalise_example(json.loads(line)))
    return Dataset.from_list(rows)


def _load_hf_dataset(name: str, split: str, limit: int) -> Dataset:
    dataset = load_dataset(name, split=split)
    if limit != -1:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset.map(normalise_example, remove_columns=dataset.column_names)


def _get_instruction_cls(instruction_id: str):
    if instruction_id in instructions_registry.INSTRUCTION_DICT:
        return instructions_registry.INSTRUCTION_DICT[instruction_id]
    if (
        rlvr_instructions_registry is not None
        and instruction_id in rlvr_instructions_registry.INSTRUCTION_DICT
    ):
        return rlvr_instructions_registry.INSTRUCTION_DICT[instruction_id]
    raise KeyError(
        f"Unknown IFBench instruction id {instruction_id!r}. "
        "Install allenai/open-instruct for IF-RLVR training constraints."
    )


def _candidate_responses(response: str, strict: bool) -> list[str]:
    if strict:
        return [response]
    lines = response.split("\n")
    response_remove_first = "\n".join(lines[1:]).strip()
    response_remove_last = "\n".join(lines[:-1]).strip()
    response_remove_both = "\n".join(lines[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        response_remove_first,
        response_remove_last,
        response_remove_both,
        response_remove_first.replace("*", ""),
        response_remove_last.replace("*", ""),
        response_remove_both.replace("*", ""),
    ]


def instruction_following_reward(
    completion, answer: str, parser: vf.Parser, strict: bool = True, **kwargs
) -> float:
    response = parser.parse_answer(completion) or ""
    payload = json.loads(answer)
    followed = []

    for instruction_id, instruction_kwargs in zip(
        payload["instruction_id_list"], payload["kwargs"]
    ):
        instruction_cls = _get_instruction_cls(instruction_id)
        instruction = instruction_cls(instruction_id)
        instruction.build_description(**instruction_kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=payload["prompt"])
        followed.append(
            any(
                candidate.strip() and instruction.check_following(candidate)
                for candidate in _candidate_responses(response, strict)
            )
        )

    if not followed:
        return 0.0
    return sum(followed) / len(followed)


def all_constraints_reward(completion, answer: str, parser: vf.Parser, strict: bool = True, **kwargs) -> float:
    return 1.0 if instruction_following_reward(completion, answer, parser, strict=strict) == 1.0 else 0.0


def load_environment(
    dataset_name: str = DEFAULT_TRAIN_DATASET,
    dataset_split: str = "train",
    eval_dataset_name: str = DEFAULT_EVAL_DATASET,
    eval_dataset_split: str = "train",
    train_jsonl: str | None = None,
    eval_jsonl: str | None = None,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    strict: bool = True,
    system_prompt: str | None = None,
) -> vf.Environment:
    def build_dataset() -> Dataset:
        if train_jsonl:
            dataset = _read_jsonl(train_jsonl)
            if num_train_examples != -1:
                dataset = dataset.select(range(min(num_train_examples, len(dataset))))
            return dataset
        return _load_hf_dataset(dataset_name, dataset_split, num_train_examples)

    def build_eval_dataset() -> Dataset:
        if eval_jsonl:
            dataset = _read_jsonl(eval_jsonl)
            if num_eval_examples != -1:
                dataset = dataset.select(range(min(num_eval_examples, len(dataset))))
            return dataset
        return _load_hf_dataset(eval_dataset_name, eval_dataset_split, num_eval_examples)

    parser = vf.Parser()
    rubric = vf.Rubric(
        funcs=[
            lambda completion, answer, **kwargs: instruction_following_reward(
                completion, answer, parser, strict=strict, **kwargs
            ),
            lambda completion, answer, **kwargs: all_constraints_reward(
                completion, answer, parser, strict=strict, **kwargs
            ),
        ],
        weights=[1.0, 0.0],
        parser=parser,
    )

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
