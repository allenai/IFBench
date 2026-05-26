"""Utilities for using IFBench as an RLVR-style training task source."""

import argparse
import json
from pathlib import Path
from typing import Literal

import evaluation_lib


RewardMode = Literal["all", "mean"]


def _load_jsonl(path: str | Path):
  with open(path, "r") as f:
    for line in f:
      if line.strip():
        yield json.loads(line)


def _write_jsonl(path: str | Path, rows):
  with open(path, "w") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def task_from_input(inp: evaluation_lib.InputExample) -> dict:
  """Converts an IFBench input into a train/eval task row."""
  return {
      "id": str(inp.key),
      "prompt": inp.prompt,
      "messages": [{"role": "user", "content": inp.prompt}],
      "instruction_id_list": inp.instruction_id_list,
      "kwargs": inp.kwargs,
      "reward_spec": {
          "type": "ifbench_instruction_following",
          "reward_modes": ["all", "mean"],
          "strict_default": True,
      },
  }


def export_tasks(input_path: str | Path, output_path: str | Path, limit: int | None = None):
  """Exports IFBench inputs as JSONL task rows with chat messages."""
  inputs = evaluation_lib.read_prompt_list(input_path)
  if limit is not None:
    inputs = inputs[:limit]
  _write_jsonl(output_path, (task_from_input(inp) for inp in inputs))
  return len(inputs)


def reward_from_follow_list(follow_instruction_list: list[bool], mode: RewardMode = "all") -> float:
  """Turns per-instruction verifier booleans into a scalar RL reward."""
  if not follow_instruction_list:
    return 0.0
  if mode == "mean":
    return sum(follow_instruction_list) / len(follow_instruction_list)
  if mode == "all":
    return float(all(follow_instruction_list))
  raise ValueError(f"Unsupported reward mode: {mode}")


def score_response(
    inp: evaluation_lib.InputExample,
    response: str | None,
    *,
    loose: bool = False,
    reward_mode: RewardMode = "all",
) -> dict:
  """Scores one model response with the IFBench instruction verifiers."""
  output = evaluation_lib.evaluate_response(inp, response or "", loose=loose)
  return {
      "key": inp.key,
      "prompt": inp.prompt,
      "response": response or "",
      "follow_all_instructions": output.follow_all_instructions,
      "follow_instruction_list": output.follow_instruction_list,
      "reward": reward_from_follow_list(output.follow_instruction_list, reward_mode),
  }


def _response_maps(response_path: str | Path):
  by_key = {}
  by_prompt = {}
  for row in _load_jsonl(response_path):
    response = row.get("response", "")
    if "key" in row:
      by_key[str(row["key"])] = response
    if "id" in row:
      by_key[str(row["id"])] = response
    if "prompt" in row:
      prompt = row["prompt"]
      by_prompt[prompt] = response
      by_prompt.setdefault(prompt.strip(), response)
  return by_key, by_prompt


def score_responses(
    input_path: str | Path,
    response_path: str | Path,
    output_path: str | Path,
    *,
    loose: bool = False,
    reward_mode: RewardMode = "all",
):
  """Scores a response JSONL and writes reward-annotated JSONL."""
  inputs = evaluation_lib.read_prompt_list(input_path)
  by_key, by_prompt = _response_maps(response_path)
  rows = []
  for inp in inputs:
    response = by_key.get(str(inp.key))
    if response is None:
      response = by_prompt.get(inp.prompt) or by_prompt.get(inp.prompt.strip())
    rows.append(
        score_response(inp, response, loose=loose, reward_mode=reward_mode)
    )
  _write_jsonl(output_path, rows)
  return len(rows)


def main():
  parser = argparse.ArgumentParser(
      description="Export or score IFBench as RLVR train/eval tasks.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  export_parser = subparsers.add_parser("export", help="Export IFBench tasks.")
  export_parser.add_argument("--input-data", required=True)
  export_parser.add_argument("--output-data", required=True)
  export_parser.add_argument("--limit", type=int)

  score_parser = subparsers.add_parser("score", help="Score model responses.")
  score_parser.add_argument("--input-data", required=True)
  score_parser.add_argument("--response-data", required=True)
  score_parser.add_argument("--output-data", required=True)
  score_parser.add_argument("--loose", action="store_true")
  score_parser.add_argument("--reward-mode", choices=["all", "mean"], default="all")

  args = parser.parse_args()
  if args.command == "export":
    count = export_tasks(args.input_data, args.output_data, limit=args.limit)
  else:
    count = score_responses(
        args.input_data,
        args.response_data,
        args.output_data,
        loose=args.loose,
        reward_mode=args.reward_mode,
    )
  print(f"Wrote {count} rows to {args.output_data}")


if __name__ == "__main__":
  main()
