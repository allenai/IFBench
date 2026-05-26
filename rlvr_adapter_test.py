import json

import evaluation_lib
import rlvr_adapter


def _first_input():
  return evaluation_lib.read_prompt_list("data/IFBench_test.jsonl")[0]


def test_export_tasks_writes_chat_messages(tmp_path):
  output_path = tmp_path / "tasks.jsonl"

  count = rlvr_adapter.export_tasks(
      "data/IFBench_test.jsonl",
      output_path,
      limit=1,
  )

  row = json.loads(output_path.read_text().strip())
  assert count == 1
  assert row["messages"] == [{"role": "user", "content": row["prompt"]}]
  assert row["reward_spec"]["type"] == "ifbench_instruction_following"
  assert row["instruction_id_list"] == ["count:keywords_multiple"]


def test_score_response_returns_all_or_mean_reward():
  inp = _first_input()
  partial_response = (
      "kaleidoscope nebula nebula whisper whisper whisper "
      "labyrinth labyrinth labyrinth labyrinth labyrinth"
  )
  complete_response = (
      f"{partial_response} "
      "paradox paradox paradox paradox paradox paradox paradox"
  )

  partial = rlvr_adapter.score_response(inp, partial_response, reward_mode="mean")
  complete = rlvr_adapter.score_response(inp, complete_response, reward_mode="all")

  assert partial["follow_instruction_list"] == [False]
  assert partial["reward"] == 0.0
  assert complete["follow_instruction_list"] == [True]
  assert complete["reward"] == 1.0


def test_score_responses_matches_prompts_with_trailing_whitespace(tmp_path):
  inp = _first_input()
  input_path = tmp_path / "input.jsonl"
  response_path = tmp_path / "responses.jsonl"
  output_path = tmp_path / "scores.jsonl"

  input_path.write_text(
      json.dumps(
          {
              "key": inp.key,
              "prompt": inp.prompt,
              "instruction_id_list": inp.instruction_id_list,
              "kwargs": inp.kwargs,
          }
      )
      + "\n"
  )
  response_path.write_text(
      json.dumps({"prompt": inp.prompt + " ", "response": "missing keywords"})
      + "\n"
  )

  count = rlvr_adapter.score_responses(input_path, response_path, output_path)

  row = json.loads(output_path.read_text().strip())
  assert count == 1
  assert row["response"] == "missing keywords"
  assert row["reward"] == 0.0
