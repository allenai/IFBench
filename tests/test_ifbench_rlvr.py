import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "environments" / "ifbench_rlvr"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ENV_DIR))


class DummyDataset(list):
    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    @property
    def column_names(self):
        return list(self[0].keys()) if self else []

    def select(self, indices):
        return DummyDataset([self[index] for index in indices])

    def map(self, function, remove_columns=None):
        return DummyDataset([function(row) for row in self])


datasets_stub = types.ModuleType("datasets")
datasets_stub.Dataset = DummyDataset
datasets_stub.load_dataset = lambda *args, **kwargs: DummyDataset()
sys.modules.setdefault("datasets", datasets_stub)

verifiers_stub = types.ModuleType("verifiers")
verifiers_stub.Environment = object
verifiers_stub.Parser = object
verifiers_stub.Rubric = lambda *args, **kwargs: object()
verifiers_stub.SingleTurnEnv = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}
sys.modules.setdefault("verifiers", verifiers_stub)

import ifbench_rlvr


class DummyParser:
    def parse_answer(self, completion):
        return completion


def test_normalise_eval_example():
    example = {
        "key": "case-1",
        "prompt": "Say hello. End with done",
        "instruction_id_list": ["last_word:last_word_answer"],
        "kwargs": [{"last_word": "done", "unused": None}],
    }

    row = ifbench_rlvr.normalise_example(example)
    answer = json.loads(row["answer"])

    assert row["question"] == "Say hello. End with done"
    assert answer["instruction_id_list"] == ["last_word:last_word_answer"]
    assert answer["kwargs"] == [{"last_word": "done"}]


def test_normalise_training_ground_truth():
    example = {
            "key": "train-1",
            "messages": [
            {
                "role": "user",
                "content": "Connect all sentences with hyphens and end with brief.",
            }
        ],
        "ground_truth": (
            "[{'instruction_id': ['detectable_format:sentence_hyphens', "
            "'last_word:last_word_answer'], 'kwargs': [None, {'last_word': 'brief'}]}]"
        ),
    }

    row = ifbench_rlvr.normalise_example(example)
    answer = json.loads(row["answer"])

    assert row["question"] == "Connect all sentences with hyphens and end with brief."
    assert answer["instruction_id_list"] == [
        "detectable_format:sentence_hyphens",
        "last_word:last_word_answer",
    ]
    assert answer["kwargs"] == [{}, {"last_word": "brief"}]


def test_instruction_reward_scores_fractional_constraints():
    row = ifbench_rlvr.normalise_example(
        {
            "key": "case-2",
            "prompt": "Include foo twice and end with done.",
            "instruction_id_list": [
                "count:numbers",
                "count:conjunctions",
            ],
            "kwargs": [
                {"N": 2},
                {"small_n": 2},
            ],
        }
    )

    assert ifbench_rlvr.instruction_following_reward(
        "There are 1 and 2 numbers.",
        row["answer"],
        DummyParser(),
    ) == 0.5
    assert ifbench_rlvr.instruction_following_reward(
        "There are 1 and 2 numbers, and this conjunction plus but makes two.",
        row["answer"],
        DummyParser(),
    ) == 1.0


def test_read_jsonl_builds_verifiers_rows(tmp_path):
    input_path = tmp_path / "ifbench.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "key": "case-3",
                "prompt": "End with done.",
                "instruction_id_list": ["last_word:last_word_answer"],
                "kwargs": [{"last_word": "done"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = ifbench_rlvr._read_jsonl(input_path)

    assert len(dataset) == 1
    assert dataset[0]["question"] == "End with done."
    assert json.loads(dataset[0]["answer"])["kwargs"] == [{"last_word": "done"}]
