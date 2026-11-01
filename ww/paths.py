"""Where results and runs live. Override with WW_RESULTS / WW_RUNS (the tests use temp dirs)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("WW_RESULTS", os.path.join(ROOT, "results"))
RUNS = os.environ.get("WW_RUNS", os.path.join(ROOT, "runs"))
