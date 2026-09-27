"""Run a synthetic one-aircraft episode from JSON files."""

import argparse
import json
from pathlib import Path

from .aircraft import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("orders", type=Path)
    parser.add_argument("--output", type=Path, help="Write replay JSON; default: stdout")
    args = parser.parse_args()
    replay = run_episode(json.loads(args.scenario.read_text()), json.loads(args.orders.read_text()))
    rendered = json.dumps(replay, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
