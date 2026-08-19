"""IFBench: verifiable instruction-following benchmark.

Exposes the verifier classes (`instructions`, `classic_instructions`),
the shared registry (`instructions_registry.INSTRUCTION_DICT`), and a
helper for locating bundled data files.
"""

from importlib import resources
from pathlib import Path

from ifbench import classic_instructions, instructions, instructions_registry, instructions_util  # noqa: F401


def data_path(name: str = "IFBench_test.jsonl") -> Path:
    """Return the filesystem path to a bundled data file."""
    return Path(resources.files(__name__) / "data" / name)
