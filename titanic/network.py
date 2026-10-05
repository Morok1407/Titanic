"""A small binary MLP with backpropagation and Adam, implemented with NumPy."""

import json

import numpy as np


class MLPClassifier:
    def __init__(self, hidden_layer_sizes=(32, 16), alpha=0.01, batch_size=64,
                 learning_rate_init=0.001, random_state=42):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.coefs_, self.intercepts_ = [], []
        self.steps = 0

    def initialize(self, n_features, n_samples):
        widths = (n_features,) + self.hidden_layer_sizes + (1,)
        self.n_samples = n_samples
        self.coefs_ = [self.rng.normal(0, np.sqrt(2 / a), size=(a, b))
                       for a, b in zip(widths[:-1], widths[1:])]
        self.intercepts_ = [np.zeros(b) for b in widths[1:]]
        self.moments = [np.zeros_like(p) for p in self.parameters()]
        self.velocities = [np.zeros_like(p) for p in self.parameters()]

    def parameters(self):
        return self.coefs_ + self.intercepts_

    def forward(self, x):
        activations, preactivations = [x], []
        for i, (weights, bias) in enumerate(zip(self.coefs_, self.intercepts_)):
            z = activations[-1] @ weights + bias
            preactivations.append(z)
            activations.append(np.maximum(z, 0) if i < len(self.coefs_) - 1
                               else np.exp(-np.logaddexp(0, -z)))
        return activations, preactivations

    def loss_and_gradients(self, x, y):
        activations, preactivations = self.forward(x)
        target = np.asarray(y).reshape(-1, 1)
        logits = preactivations[-1]
        regularization = self.alpha / self.n_samples
        loss = np.mean(np.logaddexp(0, logits) - target * logits)
        loss += 0.5 * regularization * sum(np.sum(w * w) for w in self.coefs_)
        delta = (activations[-1] - target) / len(x)
        grad_w, grad_b = [None] * len(self.coefs_), [None] * len(self.intercepts_)
        for i in range(len(self.coefs_) - 1, -1, -1):
            grad_w[i] = activations[i].T @ delta + regularization * self.coefs_[i]
            grad_b[i] = delta.sum(axis=0)
            if i:
                delta = (delta @ self.coefs_[i].T) * (preactivations[i - 1] > 0)
        return float(loss), grad_w + grad_b

    def partial_fit(self, x, y, classes=None):
        if not self.coefs_:
            self.initialize(x.shape[1], len(x))
        order = self.rng.permutation(len(x))
        for start in range(0, len(x), self.batch_size):
            batch = order[start:start + self.batch_size]
            _, gradients = self.loss_and_gradients(x[batch], y[batch])
            self.steps += 1
            for p, grad, moment, velocity in zip(self.parameters(), gradients, self.moments, self.velocities):
                moment *= 0.9
                moment += 0.1 * grad
                velocity *= 0.999
                velocity += 0.001 * grad * grad
                m_hat = moment / (1 - 0.9 ** self.steps)
                v_hat = velocity / (1 - 0.999 ** self.steps)
                p -= self.learning_rate_init * m_hat / (np.sqrt(v_hat) + 1e-8)
        return self

    def predict_proba(self, x):
        probability = self.forward(x)[0][-1].ravel()
        return np.column_stack([1 - probability, probability])


class SurvivalPipeline:
    def __init__(self, preprocessor, classifier):
        self.preprocessor = preprocessor
        self.classifier = classifier

    def predict_proba(self, features):
        return self.classifier.predict_proba(self.preprocessor.transform(features))


def save_model(path, pipeline, metadata):
    network = pipeline.classifier
    state = {**metadata, "preprocessor": pipeline.preprocessor.state(),
             "layers": list(network.hidden_layer_sizes), "alpha": network.alpha,
             "n_samples": network.n_samples, "n_layers": len(network.coefs_)}
    arrays = {f"weights_{i}": value for i, value in enumerate(network.coefs_)}
    arrays.update({f"bias_{i}": value for i, value in enumerate(network.intercepts_)})
    np.savez_compressed(path, metadata=np.array(json.dumps(state)), **arrays)


def load_model(path):
    from .features import Preprocessor
    # Numeric arrays and JSON only. Loading does not execute pickled Python code.
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        preprocessor = Preprocessor.from_state(metadata["preprocessor"])
        network = MLPClassifier(metadata["layers"], metadata["alpha"])
        network.coefs_ = [archive[f"weights_{i}"].copy() for i in range(metadata["n_layers"])]
        network.intercepts_ = [archive[f"bias_{i}"].copy() for i in range(metadata["n_layers"])]
        network.n_samples = metadata["n_samples"]
    return SurvivalPipeline(preprocessor, network), metadata


def log_loss(y, probability, labels=None):
    probability = np.clip(probability, np.finfo(float).eps, 1 - np.finfo(float).eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(probability) + (1 - y) * np.log1p(-probability)))


def accuracy_score(y, predicted):
    return float(np.mean(np.asarray(y) == np.asarray(predicted)))


def classification_metrics(y, probability):
    y = np.asarray(y)
    predicted = probability >= 0.5
    tn = int(np.sum((y == 0) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    tp = int(np.sum((y == 1) & predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    positive, negative = probability[y == 1], probability[y == 0]
    pairs = positive[:, None] - negative[None, :]
    auc = float(np.mean((pairs > 0) + 0.5 * (pairs == 0)))
    return {"roc_auc": auc, "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "brier_score": float(np.mean((probability - y) ** 2)),
            "confusion_matrix": [[tn, fp], [fn, tp]]}
