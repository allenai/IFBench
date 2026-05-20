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

"""Smoke runner for using IFBench verifiers as RLVR rewards."""

import dataclasses
import json

from absl import app
from absl import flags

import evaluation_lib
import reward_lib


_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "Path to IFBench input jsonl.", required=True
)
_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data",
    None,
    "Path to jsonl rows with prompt and response fields.",
    required=True,
)
_MODE = flags.DEFINE_enum(
    "mode", "loose", ["loose", "strict"], "IFBench verifier mode."
)
_LIMIT = flags.DEFINE_integer(
    "limit", None, "Optional maximum number of examples to score."
)
_OUTPUT_JSONL = flags.DEFINE_string(
    "output_jsonl", None, "Optional path for per-example reward results."
)


def _write_jsonl(path: str, results: list[reward_lib.RewardResult]) -> None:
  with open(path, "w") as f:
    for result in results:
      f.write(json.dumps(dataclasses.asdict(result)))
      f.write("\n")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  examples = evaluation_lib.read_prompt_list(_INPUT_DATA.value)
  if _LIMIT.value is not None:
    examples = examples[: _LIMIT.value]
  prompt_to_response = evaluation_lib.read_prompt_to_response_dict(
      _INPUT_RESPONSE_DATA.value
  )

  results = [
      reward_lib.score_response(
          example,
          prompt_to_response.get(example.prompt, ""),
          mode=_MODE.value,
      )
      for example in examples
  ]
  summary = reward_lib.summarize_results(results)
  print(json.dumps(summary, sort_keys=True))

  if _OUTPUT_JSONL.value:
    _write_jsonl(_OUTPUT_JSONL.value, results)


if __name__ == "__main__":
  app.run(main)
