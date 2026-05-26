"""Prime Intellect Verifiers environment for IFBench.

This module is optional: install the ``rlvr`` extra before loading it with
Verifiers or prime-rl.
"""

from collections.abc import Mapping
from typing import Literal

import evaluation_lib
import rlvr_adapter

try:
  import verifiers as vf
except ImportError as exc:  # pragma: no cover - exercised only without extra deps.
  raise ImportError(
      "ifbench_verifiers requires the optional rlvr dependencies. "
      "Install with `uv sync --extra rlvr` or `pip install .[rlvr]`."
  ) from exc


RewardMode = Literal["all", "mean"]


class IFBenchTasksetConfig(vf.TasksetConfig):
  input_path: str = "data/IFBench_test.jsonl"
  eval_input_path: str | None = None
  limit: int | None = None
  eval_limit: int | None = None
  loose: bool = False
  reward_mode: RewardMode = "all"
  system_prompt: str | None = None


def _input_from_task(task) -> evaluation_lib.InputExample:
  return evaluation_lib.InputExample(
      key=task.get("key", task.get("example_id", "")),
      instruction_id_list=list(task["instruction_id_list"]),
      prompt=task["question"],
      kwargs=list(task["kwargs"]),
  )


def _assistant_response(completion) -> str:
  for message in reversed(completion or []):
    if isinstance(message, Mapping):
      role = message.get("role")
      content = message.get("content")
    else:
      role = getattr(message, "role", None)
      content = getattr(message, "content", None)

    if role == "assistant":
      return str(content or "")

  return ""


class IFBenchTaskset(vf.Taskset):
  config_type = IFBenchTasksetConfig
  config: IFBenchTasksetConfig

  def __init__(self, config: IFBenchTasksetConfig | dict | None = None):
    resolved_config = self.config_type.from_config(config)
    eval_path = resolved_config.eval_input_path or resolved_config.input_path
    super().__init__(
        source=lambda: list(
            self._load_ifbench_rows(resolved_config.input_path, resolved_config.limit)
        ),
        eval_source=lambda: list(
            self._load_ifbench_rows(eval_path, resolved_config.eval_limit)
        ),
        system_prompt=resolved_config.system_prompt,
        config=resolved_config,
    )

  def _load_ifbench_rows(self, path: str, limit: int | None):
    inputs = evaluation_lib.read_prompt_list(path)
    if limit is not None:
      inputs = inputs[:limit]

    for inp in inputs:
      task = rlvr_adapter.task_from_input(inp)
      yield {
          "example_id": str(inp.key),
          "key": str(inp.key),
          "prompt": task["messages"],
          "question": inp.prompt,
          "instruction_id_list": inp.instruction_id_list,
          "kwargs": inp.kwargs,
          "info": {"reward_spec": task["reward_spec"]},
      }

  @vf.reward(weight=1.0)
  async def ifbench_reward(self, task, state) -> float:
    response = _assistant_response(state.get("completion"))
    inp = _input_from_task(task)
    scored = rlvr_adapter.score_response(
        inp,
        response,
        loose=self.config.loose,
        reward_mode=self.config.reward_mode,
    )
    return float(scored["reward"])


class IFBenchEnvConfig(vf.EnvConfig):
  taskset: IFBenchTasksetConfig = IFBenchTasksetConfig()
  harness: vf.HarnessConfig = vf.HarnessConfig()


def load_taskset(config: IFBenchTasksetConfig) -> IFBenchTaskset:
  return IFBenchTaskset(config=config)


def load_environment(config: IFBenchEnvConfig) -> vf.Env:
  return vf.Env(
      taskset=IFBenchTaskset(config=config.taskset),
      harness=vf.Harness(config=config.harness),
  )
