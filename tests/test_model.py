import json
import unittest

import numpy as np
import pandas as pd
from titanic.network import MLPClassifier, accuracy_score, classification_metrics, log_loss

from titanic.features import NUMERIC, build_features, deck_from_cabin
from titanic.predict import ROOT, SurvivalPredictor, validate_passenger
from titanic.train import make_splits


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "data/titanic.csv")
        cls.predictor = SurvivalPredictor()
        cls.indices = json.loads((ROOT / "reports/split_indices.json").read_text())

    def test_outcome_columns_cannot_change_features(self):
        altered = self.raw.copy()
        altered["survived"] = 1 - altered["survived"]
        altered["boat"] = "LEAK"
        altered["body"] = 99999
        altered["name"] = "REDACTED"
        altered["ticket"] = "REDACTED"
        altered["home.dest"] = "REDACTED"
        pd.testing.assert_frame_equal(build_features(self.raw), build_features(altered))

    def test_split_has_no_ticket_overlap_and_is_reproducible(self):
        recomputed = make_splits(self.raw, self.raw.survived.to_numpy())
        self.assertEqual(self.indices, {k: v.tolist() for k, v in recomputed.items()})
        tickets = self.raw.ticket.astype(str).str.upper().str.replace(r"\s+", "", regex=True)
        sets = {name: set(tickets.iloc[index]) for name, index in self.indices.items()}
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["test"] & sets["validation"])
        all_indices = sum(self.indices.values(), [])
        self.assertEqual(sorted(all_indices), list(range(len(self.raw))))

    def test_imputation_statistics_come_from_training_only(self):
        features = build_features(self.raw.iloc[self.indices["train"]])
        expected = features[NUMERIC].median().to_numpy()
        actual = self.predictor.pipeline.preprocessor.medians
        np.testing.assert_allclose(actual, expected)

    def test_missing_inputs_have_finite_prediction(self):
        result = self.predictor.predict({"sex": "male", "pclass": 3})
        self.assertTrue(0 <= result["survival_probability"] <= 1)
        self.assertEqual(set(result["missing_inputs"]), {"age", "fare", "deck", "embarked"})
        self.assertEqual(set(result["defaulted_inputs"]), {"sibsp", "parch"})

    def test_invalid_input_rejected(self):
        cases = [
            {"sex": "unknown"}, {"pclass": 4}, {"pclass": 1.5}, {"age": -1},
            {"age": float("nan")}, {"age": float("inf")}, {"age": True},
            {"fare": -1}, {"sibsp": 0.5}, {"parch": None}, {"deck": "Z"},
            {"embarked": "X"}, {"boat": "1"}, {"survived": 1},
        ]
        for override in cases:
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_passenger({"sex": "female", "pclass": 1, **override})

    def test_ui_feature_path_matches_training_path(self):
        sample = self.raw.sample(30, random_state=31)
        expected = self.predictor.pipeline.predict_proba(build_features(sample))[:, 1]
        for (_, row), probability in zip(sample.iterrows(), expected):
            passenger = {key: (None if pd.isna(row[key]) else row[key])
                         for key in ["sex", "pclass", "age", "fare", "sibsp", "parch", "embarked"]}
            passenger["deck"] = deck_from_cabin(row.cabin)
            actual = self.predictor.predict(passenger)["survival_probability"]
            self.assertAlmostEqual(actual, probability, places=12)

    def test_saved_model_reproduces_reported_test_scores(self):
        test = self.raw.iloc[self.indices["test"]]
        probability = self.predictor.pipeline.predict_proba(build_features(test))[:, 1]
        report = json.loads((ROOT / "reports/metrics.json").read_text())
        self.assertAlmostEqual(accuracy_score(test.survived, probability >= 0.5), report["test"]["accuracy"])
        self.assertAlmostEqual(log_loss(test.survived, probability), report["test"]["log_loss"])
        saved = pd.read_csv(ROOT / "reports/test_predictions.csv")
        np.testing.assert_allclose(probability, saved.survival_probability, atol=1e-14, rtol=0)


class NetworkMathTests(unittest.TestCase):
    def test_backpropagation_matches_finite_differences(self):
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=(7, 3)), np.array([0, 1, 0, 1, 1, 0, 1])
        network = MLPClassifier((4, 2), alpha=0.2)
        network.initialize(3, len(x))
        # ReLU has no derivative exactly at zero; test at differentiable points.
        for bias in network.intercepts_:
            bias[:] = rng.normal(0.1, 0.15, size=bias.shape)
        _, analytic = network.loss_and_gradients(x, y)
        epsilon = 1e-6
        for parameter, gradient in zip(network.parameters(), analytic):
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = network.loss_and_gradients(x, y)[0]
                parameter[index] = original - epsilon
                minus = network.loss_and_gradients(x, y)[0]
                parameter[index] = original
                self.assertAlmostEqual(gradient[index], (plus - minus) / (2 * epsilon), places=6)

    def test_adam_learns_simple_separable_problem(self):
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])
        network = MLPClassifier((), alpha=0, learning_rate_init=0.05)
        for _ in range(200):
            network.partial_fit(x, y)
        self.assertLess(log_loss(y, network.predict_proba(x)[:, 1]), 0.05)

    def test_auc_and_confusion_matrix_with_known_values(self):
        y = np.array([0, 0, 1, 1])
        metrics = classification_metrics(y, np.array([0.1, 0.4, 0.35, 0.8]))
        self.assertEqual(metrics["roc_auc"], 0.75)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertEqual(classification_metrics(y, np.full(4, 0.5))["roc_auc"], 0.5)


if __name__ == "__main__":
    unittest.main()
