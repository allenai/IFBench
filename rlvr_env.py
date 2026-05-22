"""Train-ready reward helpers for IFBench / IF-RLVR style loops.

The main evaluator scores a whole response file at once. RLVR pipelines usually
need a smaller interface: given one prompt and one sampled completion, return a
verifiable scalar reward plus per-instruction diagnostics. This module keeps
that wrapper close to the existing IFBench verifier implementation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import evaluation_lib
import instructions_registry


EvaluationMode = Literal["strict", "loose"]
RewardMode = Literal["all", "fraction"]


@dataclasses.dataclass(frozen=True)
class RewardResult:
  """Verifier output for one prompt/completion pair."""

  prompt: str
  response: str
  reward: float
  follow_all_instructions: bool
  follow_instruction_list: list[bool]
  instruction_id_list: list[str]


def _clean_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
  return {key: value for key, value in kwargs.items() if value is not None}


def _response_variants(response: str, mode: EvaluationMode) -> list[str]:
  if mode == "strict":
    return [response]

  lines = response.split("\n")
  response_remove_first = "\n".join(lines[1:]).strip()
  response_remove_last = "\n".join(lines[:-1]).strip()
  response_remove_both = "\n".join(lines[1:-1]).strip()
  revised_response = response.replace("*", "")

  return [
      response,
      revised_response,
      response_remove_first,
      response_remove_last,
      response_remove_both,
      response_remove_first.replace("*", ""),
      response_remove_last.replace("*", ""),
      response_remove_both.replace("*", ""),
  ]


def score_response(
    inp: evaluation_lib.InputExample,
    response: str | None,
    *,
    evaluation_mode: EvaluationMode = "strict",
    reward_mode: RewardMode = "all",
) -> RewardResult:
  """Score one sampled completion with IFBench verification functions.

  Args:
    inp: IFBench input example.
    response: Model completion for ``inp.prompt``.
    evaluation_mode: ``strict`` uses the exact completion; ``loose`` mirrors the
      benchmark's loose scoring variants.
    reward_mode: ``all`` returns 1.0 only when all instructions pass; ``fraction``
      returns the mean per-instruction pass rate.
  """

  if evaluation_mode not in ("strict", "loose"):
    raise ValueError(f"unsupported evaluation_mode: {evaluation_mode}")
  if reward_mode not in ("all", "fraction"):
    raise ValueError(f"unsupported reward_mode: {reward_mode}")

  response = response or ""
  response_variants = _response_variants(response, evaluation_mode)
  follow_instruction_list: list[bool] = []

  for index, instruction_id in enumerate(inp.instruction_id_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)
    instruction.build_description(**_clean_kwargs(inp.kwargs[index]))
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    follows = any(
        variant.strip() and instruction.check_following(variant)
        for variant in response_variants
    )
    follow_instruction_list.append(follows)

  follow_all = all(follow_instruction_list)
  if reward_mode == "all":
    reward = 1.0 if follow_all else 0.0
  else:
    reward = sum(follow_instruction_list) / len(follow_instruction_list)

  return RewardResult(
      prompt=inp.prompt,
      response=response,
      reward=reward,
      follow_all_instructions=follow_all,
      follow_instruction_list=follow_instruction_list,
      instruction_id_list=list(inp.instruction_id_list),
  )


def iter_reward_records(
    inputs: Iterable[evaluation_lib.InputExample],
    prompt_to_response: dict[str, str],
    *,
    evaluation_mode: EvaluationMode = "strict",
    reward_mode: RewardMode = "all",
) -> Iterable[dict[str, Any]]:
  """Yield JSON-serializable training records with verifier rewards."""

  for inp in inputs:
    response = prompt_to_response.get(inp.prompt)
    if response is None:
      response = prompt_to_response.get(inp.prompt.strip())
    result = score_response(
        inp,
        response,
        evaluation_mode=evaluation_mode,
        reward_mode=reward_mode,
    )
    yield dataclasses.asdict(result)


def write_reward_records(
    *,
    input_data: str | Path,
    input_response_data: str | Path,
    output_path: str | Path,
    evaluation_mode: EvaluationMode = "strict",
    reward_mode: RewardMode = "all",
) -> None:
  """Convert IFBench prompt/response JSONL files to reward-labeled JSONL."""

  inputs = evaluation_lib.read_prompt_list(str(input_data))
  prompt_to_response = evaluation_lib.read_prompt_to_response_dict(
      str(input_response_data)
  )
  for prompt, response in list(prompt_to_response.items()):
    prompt_to_response.setdefault(prompt.strip(), response)
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with output_path.open("w", encoding="utf-8") as output_file:
    for record in iter_reward_records(
        inputs,
        prompt_to_response,
        evaluation_mode=evaluation_mode,
        reward_mode=reward_mode,
    ):
      output_file.write(json.dumps(record, ensure_ascii=False))
      output_file.write("\n")


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Create reward-labeled IFBench JSONL records for RLVR loops."
  )
  parser.add_argument("--input_data", required=True)
  parser.add_argument("--input_response_data", required=True)
  parser.add_argument("--output_path", required=True)
  parser.add_argument(
      "--evaluation_mode", choices=("strict", "loose"), default="strict"
  )
  parser.add_argument("--reward_mode", choices=("all", "fraction"), default="all")
  args = parser.parse_args()

  write_reward_records(
      input_data=args.input_data,
      input_response_data=args.input_response_data,
      output_path=args.output_path,
      evaluation_mode=args.evaluation_mode,
      reward_mode=args.reward_mode,
  )


if __name__ == "__main__":
  main()
