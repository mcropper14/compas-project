"""One module per fairness method. Each exposes run(splits) which trains on
train, selects hyperparameters on val, and records val/test/shift rows."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
