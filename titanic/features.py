"""Features available before the disaster; no outcome-dependent information."""

import re

import numpy as np
import pandas as pd

FEATURE_VERSION = 1
NUMERIC = ["age", "fare_log", "sibsp", "parch", "family_size"]
CATEGORICAL = ["sex", "pclass", "deck", "embarked"]
DECKS = list("ABCDEFGT")


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature extraction. All fitted transforms live in the pipeline."""
    result = pd.DataFrame(index=raw.index)
    for column in ["age", "fare", "sibsp", "parch"]:
        result[column] = pd.to_numeric(raw[column], errors="raise")
    result["fare_log"] = np.log1p(result.pop("fare"))
    result["family_size"] = 1 + result["sibsp"] + result["parch"]
    result["sex"] = raw["sex"].astype(str).str.lower()
    result["pclass"] = raw["pclass"].astype(int).astype(str)
    result["deck"] = raw["cabin"].map(deck_from_cabin)
    result["embarked"] = raw["embarked"].fillna("UNKNOWN").astype(str).str.upper()
    return result[NUMERIC + CATEGORICAL]


def deck_from_cabin(value) -> str:
    if value is None or pd.isna(value):
        return "UNKNOWN"
    match = re.search(r"[A-Za-z]", str(value))
    deck = match.group().upper() if match else "UNKNOWN"
    return deck if deck in DECKS else "UNKNOWN"


class Preprocessor:
    categories = [["female", "male"], ["1", "2", "3"], DECKS + ["UNKNOWN"],
                  ["C", "Q", "S", "UNKNOWN"]]

    def numeric_values(self, features):
        values = features[NUMERIC].to_numpy(dtype=float)
        filled = np.where(np.isnan(values), self.medians, values)
        return np.column_stack([filled, np.isnan(values[:, self.missing_indices]).astype(float)])

    def fit_transform(self, features):
        values = features[NUMERIC].to_numpy(dtype=float)
        self.medians = np.nanmedian(values, axis=0)
        if not np.isfinite(self.medians).all():
            raise ValueError("A numeric training column has no usable observations")
        self.missing_indices = np.flatnonzero(np.isnan(values).any(axis=0))
        filled = self.numeric_values(features)
        self.means = filled.mean(axis=0)
        self.scales = filled.std(axis=0)
        self.scales[self.scales == 0] = 1.0
        return self.transform(features)

    def transform(self, features):
        encoded = [(self.numeric_values(features) - self.means) / self.scales]
        for column, categories in zip(CATEGORICAL, self.categories):
            values = features[column].to_numpy()
            if not np.isin(values, categories).all():
                raise ValueError(f"Unknown category in {column}")
            encoded.append((values[:, None] == np.array(categories)[None, :]).astype(float))
        result = np.column_stack(encoded)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite transformed features")
        return result

    def get_feature_names_out(self):
        return np.array(NUMERIC + [f"missing_{NUMERIC[i]}" for i in self.missing_indices]
                        + [f"{name}_{value}" for name, values in zip(CATEGORICAL, self.categories) for value in values])

    def state(self):
        return {name: getattr(self, name).tolist() for name in ["medians", "missing_indices", "means", "scales"]}

    @classmethod
    def from_state(cls, state):
        instance = cls()
        for name, value in state.items():
            setattr(instance, name, np.array(value, dtype=int if name == "missing_indices" else float))
        return instance


def make_preprocessor():
    return Preprocessor()
