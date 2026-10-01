"""Create an offline research workspace from recorded Nightjar replay artifacts.

The browser inspects recorded episodes and model descriptors. The CLI can record
explicit synthetic scenarios and a supplied PPO checkpoint before bundling HTML,
styles, scripts, and data without network dependencies.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Sequence

VIEWER_VERSION = "nightjar-viewer-0.4"
ASSET_DIRECTORY = Path(__file__).with_name("viewer_assets")


def build_viewer_html(replays: Sequence[dict], title: str = "Nightjar Research Workspace", *, models: Sequence[dict] = ()) -> str:
    """Bundle recorded replays and local UI assets into a portable HTML file.

    Replay fields are preserved verbatim. Missing scientific metadata remains
    unavailable in the UI rather than being inferred from another episode.
    """
    if not replays:
        raise ValueError("at least one replay is required to generate viewer HTML")
    if any(not isinstance(replay, dict) for replay in replays):
        raise ValueError("each replay must be a JSON object")
    if any(not isinstance(model, dict) for model in models):
        raise ValueError("each model record must be a JSON object")
    try:
        payload = json.dumps(list(replays), sort_keys=True, allow_nan=False)
        model_payload = json.dumps(list(models), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"replays must contain finite JSON values: {exc}") from exc
    # JSON embedded in a script must not be able to close its HTML element.
    payload = (payload.replace("<", "\\u003c").replace(">", "\\u003e")
               .replace("&", "\\u0026").replace("\u2028", "\\u2028")
               .replace("\u2029", "\\u2029"))
    model_payload = (model_payload.replace("<", "\\u003c").replace(">", "\\u003e")
                     .replace("&", "\\u0026").replace("\u2028", "\\u2028")
                     .replace("\u2029", "\\u2029"))
    replacements = {
        "__TITLE__": html.escape(title, quote=True),
        "__VERSION__": VIEWER_VERSION,
        "__CSS__": (ASSET_DIRECTORY / "workspace.css").read_text(encoding="utf-8"),
        "__JS__": (ASSET_DIRECTORY / "workspace.js").read_text(encoding="utf-8"),
        "__PAYLOAD__": payload,
        "__MODELS__": model_payload,
    }
    template = (ASSET_DIRECTORY / "workspace.html").read_text(encoding="utf-8")
    # One pass keeps placeholder-like strings in data and titles unchanged.
    return re.sub(r"__(?:TITLE|VERSION|CSS|JS|PAYLOAD|MODELS)__", lambda match: replacements[match[0]], template)


def write_viewer_html(replays: Sequence[dict], output_path: Path, title: str | None = None, *, models: Sequence[dict] = ()) -> None:
    """Write a portable workspace, creating the destination directory if needed."""
    content = build_viewer_html(replays, title=title if title is not None else "Nightjar Research Workspace", models=models)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="*", type=Path, help="Recorded replay JSON files")
    parser.add_argument("--demo", action="store_true", help="Replay three baselines on synthetic development case 00")
    parser.add_argument("--scenario", type=Path, help="Run baseline comparisons on an explicit integrated synthetic scenario")
    parser.add_argument("--checkpoint", type=Path, help="Inspect PPO weights and include their replay when running a scenario")
    parser.add_argument("--model-config", type=Path, help="Explicit model configuration for a raw state dictionary")
    parser.add_argument("--output", type=Path, default=Path("runs/viewer.html"), help="Destination HTML file")
    parser.add_argument("--title", help="Custom document title")
    args = parser.parse_args()
    if sum((bool(args.demo), bool(args.scenario), bool(args.replays))) != 1:
        parser.error("provide exactly one of replay files, --demo or --scenario")
    if args.model_config and not args.checkpoint:
        parser.error("--model-config requires --checkpoint")
    try:
        models = []
        controller = controller_record = None
        if args.checkpoint:
            from .viewer_models import load_controller_checkpoint
            config = json.loads(args.model_config.read_text()) if args.model_config else None
            controller, controller_record = load_controller_checkpoint(args.checkpoint, config)
            models.append(controller_record)
        if args.demo or args.scenario:
            from .discrimination_family import build_scenarios
            from .integrated import run_integrated
            from .viewer_models import physical_model_record, controller_replay

            if args.demo:
                development, _, _ = build_scenarios()
                scenario = development[0]
            else:
                scenario = json.loads(args.scenario.read_text())
            if not isinstance(scenario, dict) or scenario.get("schema_version") != "integrated-synthetic-0.1":
                raise ValueError("--scenario requires an integrated-synthetic-0.1 JSON object")
            replays = [run_integrated(scenario, policy) for policy in
                       ("heuristic", "containment_heuristic", "no_intervention")]
            physical = physical_model_record(scenario)
            models.append(physical)
            if controller is not None:
                replays.insert(0, controller_replay(controller, controller_record, scenario))
            for replay in replays:
                replay["physical_model_id"] = physical["id"]
                if args.demo:
                    replay["workspace_split"] = "development"
        else:
            replays = [json.loads(path.read_text(encoding="utf-8")) for path in args.replays]
        write_viewer_html(replays, args.output, title=args.title, models=models)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Generated Nightjar Research Workspace at: {args.output}")


if __name__ == "__main__":
    main()
