import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from merge_geotiff.cli import build_parser, cli, parse_args, validate_args
from merge_geotiff.processing import main

__all__ = ["build_parser", "cli", "main", "parse_args", "validate_args"]


if __name__ == "__main__":
    cli()
