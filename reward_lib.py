# coding=utf-8
# Copyright 2026 Allen Institute for AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward helpers for using IFBench verifiers in RLVR training loops."""

import copy
import dataclasses
from typing import Callable, Iterable, Mapping, Sequence

import evaluation_lib


@dataclasses.dataclass(frozen=True)
class RewardResult:
  """Structured reward output for a single prompt/response pair."""

  instruction_id_list: list[str]
  prompt: str
  response: str
  mode: str
  prompt_reward: float
  instruction_reward: float
  follow_instruction_list: list[bool]


def _as_input_example(example):
  """Converts a json-like example into an InputExample if needed."""
  if isinstance(example, evaluation_lib.InputExample):
    return example
  return evaluation_lib.InputExample(
      key=example["key"],
      instruction_id_list=example["instruction_id_list"],
      prompt=example["prompt"],
      kwargs=example["kwargs"],
  )


def _copy_input_example(example):
  """Copies an input example and normalizes nullable kwargs before scoring."""
  example = _as_input_example(example)
  return evaluation_lib.InputExample(
      key=example.key,
      instruction_id_list=list(example.instruction_id_list),
      prompt=example.prompt,
      kwargs=[
          {key: value for key, value in kwargs.items() if value is not None}
          for kwargs in copy.deepcopy(example.kwargs)
      ],
  )


def score_response(example, response: str, mode: str = "loose") -> RewardResult:
  """Scores one response with IFBench verifiers.

  Args:
    example: An evaluation_lib.InputExample or json-like dict from IFBench data.
    response: Model response to score.
    mode: Either "loose" or "strict", matching IFBench evaluation modes.

  Returns:
    RewardResult with a binary prompt_reward and fractional instruction_reward.
  """
  scoring_functions = {
      "loose": evaluation_lib.test_instruction_following_loose,
      "strict": evaluation_lib.test_instruction_following_strict,
  }
  if mode not in scoring_functions:
    raise ValueError(f"mode must be one of {sorted(scoring_functions)}")

  inp = _copy_input_example(example)
  output = scoring_functions[mode](inp, {inp.prompt: response})
  followed_count = sum(output.follow_instruction_list)
  instruction_count = len(output.follow_instruction_list)
  instruction_reward = (
      followed_count / instruction_count if instruction_count else 0.0
  )

  return RewardResult(
      instruction_id_list=list(output.instruction_id_list),
      prompt=output.prompt,
      response=output.response,
      mode=mode,
      prompt_reward=1.0 if output.follow_all_instructions else 0.0,
      instruction_reward=instruction_reward,
      follow_instruction_list=list(output.follow_instruction_list),
  )


def build_prompt_index(examples: Iterable) -> dict[str, evaluation_lib.InputExample]:
  """Builds a prompt-to-example index for reward functions."""
  return {example.prompt: example for example in map(_as_input_example, examples)}


def make_reward_fn(
    examples: Iterable,
    mode: str = "loose",
    *,
    missing_prompt_reward: float = 0.0,
) -> Callable[[Sequence[str], Sequence[str]], list[float]]:
  """Creates a batch reward function for RLVR trainers.

  The returned function accepts parallel prompt and response batches and returns
  one binary prompt-level reward per pair. Unknown prompts receive
  missing_prompt_reward so streaming trainers can continue safely.
  """
  prompt_index = build_prompt_index(examples)

  def reward_fn(prompts: Sequence[str], responses: Sequence[str]) -> list[float]:
    if len(prompts) != len(responses):
      raise ValueError("prompts and responses must have the same length")

    rewards = []
    for prompt, response in zip(prompts, responses):
      example = prompt_index.get(prompt)
      if example is None:
        rewards.append(missing_prompt_reward)
      else:
        rewards.append(score_response(example, response, mode).prompt_reward)
    return rewards

  return reward_fn


def score_response_batch(
    prompt_index: Mapping[str, evaluation_lib.InputExample],
    prompts: Sequence[str],
    responses: Sequence[str],
    mode: str = "loose",
) -> list[RewardResult]:
  """Scores a batch and returns structured per-example results."""
  if len(prompts) != len(responses):
    raise ValueError("prompts and responses must have the same length")

  results = []
  for prompt, response in zip(prompts, responses):
    if prompt not in prompt_index:
      raise KeyError(f"prompt not found in IFBench inputs: {prompt!r}")
    results.append(score_response(prompt_index[prompt], response, mode))
  return results
