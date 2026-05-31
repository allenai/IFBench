# ifbench-rlvr

Verifiers environment for IFBench instruction-following RLVR.

## Overview

- Environment ID: `ifbench-rlvr`
- Task type: single-turn instruction following
- Reward: fraction of IFBench verifier constraints satisfied by the model response
- Primary training dataset: `allenai/IF_multi_constraints_upto5`
- Default eval dataset: `allenai/IFBench_test`

The environment converts IFBench examples into `verifiers` single-turn tasks. Each
task prompt is the IFBench user prompt, and the reward calls the same verifier
classes used by IFBench. The released IF-RLVR training split references the
additional verifier registry from `allenai/open-instruct`, so that package is
declared as an environment dependency.

## Quickstart

From this directory:

```bash
prime eval run . -m openai/gpt-5-nano -n 20 -r 1
```

Use a small local smoke test with the packaged IFBench sample data:

```bash
python -m pytest tests/test_ifbench_rlvr.py
```

## Environment Arguments

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_name` | str | `allenai/IF_multi_constraints_upto5` | Hugging Face train dataset. |
| `dataset_split` | str | `train` | Train split. |
| `eval_dataset_name` | str | `allenai/IFBench_test` | Hugging Face eval dataset. |
| `eval_dataset_split` | str | `train` | Eval split. |
| `train_jsonl` | str or null | null | Optional local IFBench-format train JSONL. |
| `eval_jsonl` | str or null | null | Optional local IFBench-format eval JSONL. |
| `num_train_examples` | int | `-1` | Limit train examples; `-1` means all. |
| `num_eval_examples` | int | `-1` | Limit eval examples; `-1` means all. |
| `strict` | bool | `true` | Use strict IFBench checking. If false, use loose checking. |
