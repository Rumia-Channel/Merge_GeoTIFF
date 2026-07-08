import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from merge_geotiff.gui import main, main_func

__all__ = ["main", "main_func"]


if __name__ == "__main__":
    main_func()
