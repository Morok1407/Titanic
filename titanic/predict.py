"""Load the saved pipeline once and call predictor.predict(passenger) from a UI."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .features import DECKS, FEATURE_VERSION, build_features
from .network import load_model

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_FIELDS = {"sex", "pclass", "age", "sibsp", "parch", "fare", "deck", "embarked"}


def numeric(value, field, minimum, maximum, *, optional=False, integer=False):
    if optional and (value is None or value == ""):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field}: expected a number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: expected a number") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{field}: expected a value from {minimum} to {maximum}")
    if integer and not number.is_integer():
        raise ValueError(f"{field}: expected an integer")
    return int(number) if integer else number


def validate_passenger(passenger: dict) -> dict:
    if not isinstance(passenger, dict):
        raise ValueError("Passenger must be a JSON object")
    extra = set(passenger) - ALLOWED_FIELDS
    if extra:
        raise ValueError(f"Unknown fields: {', '.join(sorted(extra))}")
    sex = str(passenger.get("sex", "")).strip().lower()
    if sex not in {"female", "male"}:
        raise ValueError("sex: use female or male")
    deck = str(passenger.get("deck") or "UNKNOWN").strip().upper()
    if deck not in DECKS + ["UNKNOWN"]:
        raise ValueError("deck: use A–G, T or UNKNOWN")
    embarked = str(passenger.get("embarked") or "UNKNOWN").strip().upper()
    if embarked not in {"C", "Q", "S", "UNKNOWN"}:
        raise ValueError("embarked: use C, Q, S or UNKNOWN")
    return {
        "sex": sex,
        "pclass": numeric(passenger.get("pclass"), "pclass", 1, 3, integer=True),
        "age": numeric(passenger.get("age"), "age", 0, 100, optional=True),
        "fare": numeric(passenger.get("fare"), "fare", 0, 1000, optional=True),
        "sibsp": numeric(passenger.get("sibsp", 0), "sibsp", 0, 20, integer=True),
        "parch": numeric(passenger.get("parch", 0), "parch", 0, 20, integer=True),
        "cabin": None if deck == "UNKNOWN" else deck,
        "embarked": None if embarked == "UNKNOWN" else embarked,
    }


class SurvivalPredictor:
    def __init__(self, model_path=ROOT / "models/titanic_mlp.npz"):
        pipeline, bundle = load_model(model_path)
        if bundle["feature_version"] != FEATURE_VERSION:
            raise ValueError("Model and feature extraction versions do not match")
        self.pipeline = pipeline
        self.threshold = bundle["threshold"]

    def predict(self, passenger: dict) -> dict:
        raw = validate_passenger(passenger)
        features = build_features(pd.DataFrame([raw]))
        probability = float(self.pipeline.predict_proba(features)[0, 1])
        return {
            "survival_probability": probability,
            "predicted_survived": int(probability >= self.threshold),
            "threshold": self.threshold,
            "missing_inputs": [key for key in ("age", "fare") if raw[key] is None]
                + (["deck"] if raw["cabin"] is None else [])
                + (["embarked"] if raw["embarked"] is None else []),
            "defaulted_inputs": [key for key in ("sibsp", "parch") if key not in passenger],
        }


def main():
    parser = argparse.ArgumentParser(description="Titanic neural network prediction")
    parser.add_argument("--input", type=Path, required=True, help="Passenger JSON file")
    parser.add_argument("--model", type=Path, default=ROOT / "models/titanic_mlp.npz")
    args = parser.parse_args()
    try:
        passenger = json.loads(args.input.read_text(encoding="utf-8-sig"))
        prediction = SurvivalPredictor(args.model).predict(passenger)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(prediction, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
