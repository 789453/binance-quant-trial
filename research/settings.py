from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from research.factor_combo_pipeline import SELECTED_HASHES


DEFAULTS = {
    "selected_hashes": SELECTED_HASHES,
    "walk_forward": {"train_months": 6, "step_months": 1, "fee_bps": 4.0},
    "ridge": {"alphas": [1.0, 10.0, 50.0, 200.0], "sample_half_life_days": 90.0},
    "lightgbm": {
        "objective": "huber", "alpha": 0.85, "learning_rate": 0.025,
        "num_leaves": 7, "max_depth": 3, "min_child_samples": 400,
        "reg_alpha": 0.5, "reg_lambda": 5.0,
    },
    "execution": {
        "z_window": 336, "smooth_span": 8, "neutral_zone": 0.20,
        "position_deadband": 0.004, "adaptive_lookback_days": 90,
        "adaptive_max_model_weight": 0.25,
    },
}


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_settings(path: str | Path | None = None) -> dict:
    settings = deepcopy(DEFAULTS)
    if path:
        override = json.loads(Path(path).read_text(encoding="utf-8"))
        _merge(settings, override)
    hashes = settings["selected_hashes"]
    if not 4 <= len(hashes) <= 8 or len(hashes) != len(set(hashes)):
        raise ValueError("selected_hashes must contain 4-8 unique factor hashes")
    return settings
