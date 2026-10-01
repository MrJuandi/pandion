"""Bind an offline workspace to recorded physical inputs and PPO checkpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any

MODEL_SCHEMA_VERSION = "nightjar-viewer-model-0.1"
PACKAGE_ROOT = Path(__file__).parent


def source_fingerprints(names: tuple[str, ...]) -> list[dict]:
    return [{"file": f"nightjar/{name}", "sha256": sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest()}
            for name in names]


def scenario_hash(scenario: dict) -> str:
    return sha256(json.dumps(scenario, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def physical_model_record(scenario: dict) -> dict:
    """Describe only the supplied integrated synthetic scenario and current code."""
    if scenario.get("schema_version") != "integrated-synthetic-0.1":
        raise ValueError("physical model inspection requires an integrated-synthetic-0.1 scenario")
    fire, execution = scenario["fire"], scenario["aircraft"]
    aircraft = execution["aircraft"]
    moisture = [float(value) for line in fire["dead_fuel_moisture_fraction"] for value in line]
    terrain = [float(value) for line in fire["elevation_m"] for value in line]
    digest = scenario_hash(scenario)
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "id": f"physical-{digest}", "role": "physical", "name": "FM1 Rothermel · integrated surface fire",
        "scenario_sha256": digest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Exploratory synthetic model; physical validation remains open.",
        "config": {"grid_rows": fire["rows"], "grid_cols": fire["cols"],
                   "cell_size_m": fire["cell_size_m"], "horizon_s": fire["horizon_s"]},
        "source_files": source_fingerprints(("rothermel.py", "fire_rothermel.py", "integrated.py", "aircraft.py", "observation.py")),
        "details": [
            {"label": "Scenario", "value": scenario["scenario_id"]},
            {"label": "Scenario seed", "value": scenario["seed"]},
            {"label": "Provenance", "value": scenario.get("provenance")},
            {"label": "CRS", "value": scenario.get("crs")},
            {"label": "Time origin UTC", "value": scenario.get("time_origin_utc")},
            {"label": "Recorded scenario units", "value": scenario.get("units")},
            {"label": "Engine field units", "value": {"time": "s", "distance": "m", "area": "ha", "water": "L"}},
            {"label": "Fuel codes", "value": sorted({str(value) for line in fire["fuel_model"] for value in line})},
            {"label": "Spread calculation", "value": "Rothermel/Albini one-size dead-fuel FM1; eight-neighbour grid-arrival coupling"},
            {"label": "Midflame wind to [east, north] (m/s)", "value": fire["midflame_wind_to_m_s"]},
            {"label": "Dead-fuel moisture fraction range", "value": [min(moisture), max(moisture)]},
            {"label": "Recorded elevation range (m)", "value": [min(terrain), max(terrain)]},
            {"label": "Decision interval (s)", "value": execution["decision_interval_s"]},
            {"label": "Loaded ground-speed proxy (m/s)", "value": aircraft["loaded_ground_speed_m_s"]},
            {"label": "Empty ground-speed proxy (m/s)", "value": aircraft["empty_ground_speed_m_s"]},
            {"label": "Initial water (L)", "value": aircraft["initial_water_l"]},
            {"label": "Refill-time proxy (s)", "value": aircraft["refill_time_s"]},
            {"label": "Water-effect radius (m)", "value": scenario["water_effect"]["radius_m"]},
            {"label": "Water-effect assumption", "value": "Permanent blocking of not-yet-reached cells; provisional unsourced software-test rule"},
            {"label": "Observation schedule", "value": scenario["observation_schedule"]},
        ],
    }


def load_controller_checkpoint(path: Path, config_override: dict | None = None) -> tuple[Any, dict]:
    """Load tensor weights on CPU, requiring explicit grid configuration.

    Raw geographic state dictionaries require a separate explicit configuration;
    dimensions and action semantics are not recovered from ambiguous tensor shapes.
    """
    try:
        import torch
        from .model import SpatialActorCritic, TACTICAL_ACTION_NAMES
    except ImportError as exc:
        raise ValueError("checkpoint inspection requires PyTorch; install requirements-rl.txt") from exc
    path = Path(path)
    data = path.read_bytes()
    try:
        saved = torch.load(BytesIO(data), weights_only=True, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"cannot load tensor checkpoint: {exc}") from exc
    if not isinstance(saved, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    wrapped = "model_state_dict" in saved
    weights = saved.get("model_state_dict") if wrapped else saved
    config = config_override if config_override is not None else saved.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no model_config; supply --model-config with explicit grid and action configuration")
    config = dict(config)
    for key in ("grid_rows", "grid_cols"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"model_config.{key} must be a positive integer")
    if config.get("action_mode") not in ("tactical", "spatial"):
        raise ValueError("model_config.action_mode must explicitly be tactical or spatial")
    if type(config.get("recurrent")) is not bool:
        raise ValueError("model_config.recurrent must explicitly be boolean")
    if not isinstance(weights, dict) or not weights or any(not isinstance(v, torch.Tensor) for v in weights.values()):
        raise ValueError("state dictionary must contain only tensors")
    if any(not bool(torch.isfinite(value).all()) for value in weights.values()):
        raise ValueError("checkpoint tensors must all be finite")
    try:
        dimensions = {"in_channels": int(weights["spatial_cnn.0.weight"].shape[1]),
                      "vector_dim": int(weights["vector_mlp.0.weight"].shape[1]),
                      "hidden_dim": int(weights["fusion.0.weight"].shape[0]),
                      "action_dim": int(weights["actor_head.weight"].shape[0])}
        if config["recurrent"]:
            dimensions["lstm_hidden_dim"] = int(weights["lstm.weight_hh"].shape[1])
    except (KeyError, IndexError, AttributeError) as exc:
        raise ValueError("checkpoint does not match SpatialActorCritic tensor structure") from exc
    for key, value in dimensions.items():
        if key in config and config[key] != value:
            raise ValueError(f"model_config.{key} conflicts with checkpoint tensors")
        config[key] = value
    expected_actions = len(TACTICAL_ACTION_NAMES) if config["action_mode"] == "tactical" else 2 + config["grid_rows"] * config["grid_cols"]
    if config["action_dim"] != expected_actions:
        raise ValueError("checkpoint action dimension does not match current action mapping")
    constructor = {key: config[key] for key in ("grid_rows", "grid_cols", "in_channels", "vector_dim", "hidden_dim", "recurrent", "action_dim")}
    if config["recurrent"]:
        constructor["lstm_hidden_dim"] = config["lstm_hidden_dim"]
    try:
        with torch.random.fork_rng(devices=[]):
            model = SpatialActorCritic(**constructor)
        model.load_state_dict(weights, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"checkpoint architecture does not match current model: {exc}") from exc
    model.action_mode = config["action_mode"]
    model.eval()
    digest = sha256(data).hexdigest()
    record = {
        "schema_version": MODEL_SCHEMA_VERSION, "id": f"controller-{digest}", "role": "controller",
        "name": "Recurrent PPO" if config["recurrent"] else "Feedforward PPO",
        "architecture": "SpatialActorCritic", "config": config,
        "checkpoint": {"filename": path.name, "sha256": digest, "bytes": len(data)},
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "tensor_count": len(weights), "selected_seed": saved.get("selected_seed") if wrapped else None,
        "training_config": saved.get("training_config") if wrapped else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": source_fingerprints(("model.py", "training_env.py", "integrated.py")),
        "status": "Weights loaded on CPU. Architecture checked; loading alone does not establish policy quality.",
        "details": [{"label": "Action names", "value": list(TACTICAL_ACTION_NAMES) if config["action_mode"] == "tactical" else "HOLD, REFILL and a local cell target"},
                    {"label": "Runtime", "value": f"PyTorch {torch.__version__}; CPU; deterministic action selection"},
                    {"label": "Checkpoint format", "value": "Model/config bundle" if wrapped else "Raw state dictionary with supplied configuration"}],
    }
    return model, record


def controller_replay(model: Any, record: dict, scenario: dict) -> dict:
    """Record one deterministic synthetic episode; never train or evaluate a split."""
    import torch
    from .model import ObservationTensorizer
    from .training_env import IntegratedTrainingEnv

    config = record["config"]
    fire = scenario["fire"]
    if (fire["rows"], fire["cols"]) != (config["grid_rows"], config["grid_cols"]):
        raise ValueError("scenario grid does not match checkpoint model_config")
    if config["in_channels"] != ObservationTensorizer.NUM_RASTER_CHANNELS or config["vector_dim"] != ObservationTensorizer.NUM_VECTOR_FEATURES:
        raise ValueError("checkpoint input dimensions do not match the current observation tensorizer")
    env = IntegratedTrainingEnv(scenario, action_mode=config["action_mode"])
    observation = env.reset(seed=scenario["seed"])
    mapper, tensorizer = env.action_mapper(), env.observation_tensorizer()
    done, state = False, None
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.inference_mode():
            while not done:
                if config["recurrent"]:
                    action, state = model.select_action(observation, tensorizer, mapper, deterministic=True, state=state)
                else:
                    action = model.select_action(observation, tensorizer, mapper, deterministic=True)
                observation, _, done, _ = env.step_discrete(action)
    finally:
        torch.set_num_threads(previous_threads)
    replay = env.replay
    replay["policy"] = "ppo"
    replay["controller_id"] = record["id"]
    replay["checkpoint_sha256"] = record["checkpoint"]["sha256"]
    return replay
