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

"""Tests for reward_lib.py."""

from absl.testing import absltest

import evaluation_lib
import reward_lib


class RewardLibTest(absltest.TestCase):

  def _example(self, instruction_ids=None, kwargs=None):
    return evaluation_lib.InputExample(
        key=1,
        instruction_id_list=instruction_ids or ["sentence:keyword"],
        prompt="Write one sentence.",
        kwargs=kwargs or [{"word": "giraffe", "N": 1}],
    )

  def test_score_response_returns_prompt_and_instruction_rewards(self):
    result = reward_lib.score_response(
        self._example(), "A giraffe walks carefully."
    )

    self.assertEqual(result.prompt_reward, 1.0)
    self.assertEqual(result.instruction_reward, 1.0)
    self.assertEqual(result.follow_instruction_list, [True])

  def test_score_response_reports_partial_instruction_reward(self):
    result = reward_lib.score_response(
        self._example(
            instruction_ids=["sentence:keyword", "count:numbers"],
            kwargs=[{"word": "giraffe", "N": 1}, {"N": 2}],
        ),
        "A giraffe walks carefully.",
    )

    self.assertEqual(result.prompt_reward, 0.0)
    self.assertEqual(result.instruction_reward, 0.5)
    self.assertEqual(result.follow_instruction_list, [True, False])

  def test_make_reward_fn_scores_known_prompts_and_handles_unknown(self):
    reward_fn = reward_lib.make_reward_fn(
        [self._example()], missing_prompt_reward=-1.0
    )

    rewards = reward_fn(
        ["Write one sentence.", "Unknown prompt."],
        ["A giraffe walks carefully.", "Anything."],
    )

    self.assertEqual(rewards, [1.0, -1.0])

  def test_score_response_filters_none_kwargs_for_loose_scoring(self):
    example = self._example(
        instruction_ids=["sentence:keyword"],
        kwargs=[{"word": "giraffe", "N": 1, "unused": None}],
    )

    result = reward_lib.score_response(example, "A giraffe walks carefully.")

    self.assertEqual(result.prompt_reward, 1.0)

  def test_batch_helpers_validate_lengths(self):
    reward_fn = reward_lib.make_reward_fn([self._example()])

    with self.assertRaisesRegex(ValueError, "same length"):
      reward_fn(["Write one sentence."], [])


if __name__ == "__main__":
  absltest.main()
