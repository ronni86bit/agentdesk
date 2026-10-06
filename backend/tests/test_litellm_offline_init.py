"""Regression: LiteLLM/CrewAI initialization must not need the network.

litellm/__init__.py fetches the remote model-cost map (httpx.get, TLS) at
import time unless LITELLM_LOCAL_MODEL_COST_MAP is already "true" in the
environment. backend/agents/__init__.py guarantees that flag via
os.environ.setdefault BEFORE any agents.* module imports crewai/litellm.

This test runs a fresh subprocess with:
  - cwd in a neutral temp dir (so no project .env can be discovered), and
  - the env var explicitly stripped,
then poisons httpx.get and imports agents -> litellm -> crewai. Any remote
model-cost-map fetch attempt fails the test; the bundled local map must be
used instead (source == "local", is_env_forced is True).
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent(
    """
    import httpx

    def _boom(*args, **kwargs):
        raise AssertionError(
            "remote model-cost-map fetch attempted during initialization"
        )

    httpx.get = _boom

    import os

    os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)

    import agents  # noqa: E402 - must set the flag before litellm is imported

    assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true", (
        "agents package did not set LITELLM_LOCAL_MODEL_COST_MAP"
    )

    import litellm  # noqa: E402

    assert litellm.model_cost, "litellm.model_cost must be populated locally"

    from litellm.litellm_core_utils.get_model_cost_map import (
        get_model_cost_map_source_info,
    )  # noqa: E402

    info = get_model_cost_map_source_info()
    assert info["source"] == "local", info
    assert info["is_env_forced"] is True, info
    assert info["url"] is None, info

    from crewai import LLM  # noqa: F401,E402 - full CrewAI import path

    print("OK")
    """
)


def test_crewai_litellm_init_requires_no_remote_model_cost_map():
    env = dict(os.environ)
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    env["PYTHONPATH"] = str(BACKEND_DIR)
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(Path(os.environ.get("TEMP", tempfile_dir()))),
        env=env,
        check=False,  # return code asserted explicitly below with context
    )
    assert proc.returncode == 0, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "OK" in proc.stdout


def tempfile_dir() -> str:
    import tempfile

    return tempfile.gettempdir()
