"""AgentDesk agents package.

This package __init__ runs BEFORE any agents.* module body (planner, writer,
critic all import crewai at module top). LiteLLM fetches its model-cost map
from the network inside `litellm/__init__.py` unless this environment flag is
already set at litellm import time — `config.py`'s load_dotenv() runs too
late for that, so the default is guaranteed here instead. Setting it to the
bundled local map removes a startup network dependency (TLS fetch that hangs
or fails on flaky networks) without changing any model behavior.
"""

import os

# Must be set before litellm is imported (CrewAI imports it); documented
# LiteLLM switch: use the packaged model_prices_and_context_window_backup.json
# instead of fetching the remote map during initialization.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
