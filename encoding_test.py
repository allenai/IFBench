"""UTF-8 JSONL I/O works under legacy Windows cp1252 locale."""
import json
import tempfile
import unittest
from pathlib import Path

import evaluation_lib


class JsonlEncodingTests(unittest.TestCase):
  def test_read_prompt_list_utf8(self):
    with tempfile.TemporaryDirectory() as temp:
      source = Path(temp) / "input.jsonl"
      prompt = "What is the capital of Kyiv? 🧪 東京"
      row = {
          "key": "utf8-1",
          "instruction_id_list": [],
          "prompt": prompt,
          "kwargs": [],
      }
      source.write_text(
          json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
      records = evaluation_lib.read_prompt_list(str(source))
      self.assertEqual(len(records), 1)
      self.assertEqual(records[0].prompt, prompt)

  def test_read_prompt_to_response_dict_utf8(self):
    with tempfile.TemporaryDirectory() as temp:
      source = Path(temp) / "answers.jsonl"
      prompt, answer = "東京の質問 🧪", "Привет, мир 🌐"
      source.write_text(
          json.dumps(
              {"prompt": prompt, "response": answer},
              ensure_ascii=False) + "\n",
          encoding="utf-8")
      records = evaluation_lib.read_prompt_to_response_dict(str(source))
      self.assertEqual(records[prompt], answer)

  def test_write_outputs_utf8_roundtrip(self):
    with tempfile.TemporaryDirectory() as temp:
      output = Path(temp) / "graded.jsonl"
      answer = "日本語 по-русски 🧪"
      item = evaluation_lib.OutputExample(
          instruction_id_list=[],
          prompt="Question",
          response=answer,
          follow_all_instructions=True,
          follow_instruction_list=[])
      evaluation_lib.write_outputs(str(output), [item])
      result = json.loads(output.read_text(encoding="utf-8"))
      self.assertEqual(result["response"], answer)


if __name__ == "__main__":
  unittest.main()
