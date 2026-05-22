import json
from pathlib import Path

import evaluation_lib
from rlvr_env import score_response
from rlvr_env import write_reward_records


ROOT = Path(__file__).resolve().parents[1]


def test_score_response_returns_scalar_reward_and_instruction_diagnostics():
  inputs = evaluation_lib.read_prompt_list(ROOT / "data" / "IFBench_test.jsonl")
  prompt_to_response = evaluation_lib.read_prompt_to_response_dict(
      ROOT / "data" / "sample_output.jsonl"
  )

  result = score_response(inputs[0], prompt_to_response.get(inputs[0].prompt))

  assert result.prompt == inputs[0].prompt
  assert 0.0 <= result.reward <= 1.0
  assert len(result.follow_instruction_list) == len(inputs[0].instruction_id_list)
  assert result.instruction_id_list == inputs[0].instruction_id_list


def test_fraction_reward_gives_partial_credit_for_multi_instruction_prompts():
  inp = evaluation_lib.InputExample(
      key=0,
      prompt="Say hello. Include keyword apple once and keyword banana twice.",
      instruction_id_list=["sentence:keyword", "sentence:keyword"],
      kwargs=[
          {"word": "apple", "N": 1},
          {"word": "banana", "N": 1},
      ],
  )

  result = score_response(
      inp,
      "hello apple here. banana appears later.",
      reward_mode="fraction",
  )

  assert result.follow_instruction_list == [True, False]
  assert result.reward == 0.5


def test_write_reward_records_outputs_jsonl(tmp_path):
  output_path = tmp_path / "rewards.jsonl"

  write_reward_records(
      input_data=ROOT / "data" / "IFBench_test.jsonl",
      input_response_data=ROOT / "data" / "sample_output.jsonl",
      output_path=output_path,
      evaluation_mode="loose",
      reward_mode="fraction",
  )

  first_record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
  assert "prompt" in first_record
  assert "response" in first_record
  assert "reward" in first_record
  assert "follow_instruction_list" in first_record
  assert first_record["response"]
