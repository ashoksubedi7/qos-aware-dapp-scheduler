import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "src"
SIMULATOR = ROOT / "simulator"
SIMULATOR_LIB = SIMULATOR / "lib"

for path in (
    SRC,
    SIMULATOR,
    SIMULATOR_LIB,
):
    path_str = str(path)

    if path_str not in sys.path:
        sys.path.insert(
            0,
            path_str,
        )
