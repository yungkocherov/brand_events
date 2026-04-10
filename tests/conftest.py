import sys
from pathlib import Path

# Make the project root importable so `from app...` works regardless of the
# directory pytest was launched from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
