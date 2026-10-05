"""Train a small MLP; select by validation loss and evaluate on a held-out test."""

import copy
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_VERSION, build_features, make_preprocessor
from .network import MLPClassifier, SurvivalPipeline, accuracy_score, classification_metrics, load_model, log_loss, save_model

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
MAX_EPOCHS = 600
PATIENCE = 40
MIN_DELTA = 1e-5
# Declared before observing the test set. No test-driven parameter tuning.
CANDIDATES = [((16,), 0.01), ((32, 16), 0.01), ((32, 16), 0.1)]


def evaluate(y, probability):
    predicted = (probability >= 0.5).astype(int)
    accuracy = float(accuracy_score(y, predicted))
    n, z = len(y), 1.96
    denominator = 1 + z * z / n
    center = (accuracy + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) / denominator
    return {
        "n": n,
        "accuracy": accuracy,
        "accuracy_wilson_95": [center - radius, center + radius],
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        **classification_metrics(y, probability),
    }


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_splits(raw, y):
    # Shared bookings must not leak across train/validation/test.
    groups = raw["ticket"].fillna("").astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    groups = groups.where(groups.ne(""), "ROW_" + raw.index.astype(str))
    # Greedy class balancing at ticket-group level. Outcome is used only for stratification.
    rng = np.random.default_rng(SEED)
    members = list(raw.groupby(groups, sort=True).indices.values())
    rng.shuffle(members)
    members.sort(key=len, reverse=True)
    totals = np.bincount(y, minlength=2)
    counts = np.zeros((5, 2))
    folds = [[] for _ in range(5)]
    for member in members:
        group_counts = np.bincount(y[member], minlength=2)
        scores = []
        for fold in range(5):
            candidate = counts.copy()
            candidate[fold] += group_counts
            scores.append(np.std(candidate / totals, axis=0).mean())
        destination = int(np.argmin(scores))
        folds[destination].extend(member.tolist())
        counts[destination] += group_counts
    folds = [np.array(fold, dtype=int) for fold in folds]
    indices = {"train": np.concatenate(folds[:3]), "validation": folds[3], "test": folds[4]}
    for name, index in indices.items():
        if len(np.unique(y[index])) != 2:
            raise ValueError(f"{name} split must contain both classes")
    assert len(set(np.concatenate(list(indices.values())))) == len(raw)
    group_sets = {name: set(groups.iloc[index]) for name, index in indices.items()}
    assert not group_sets["train"] & group_sets["validation"]
    assert not group_sets["train"] & group_sets["test"]
    assert not group_sets["validation"] & group_sets["test"]
    return indices


def fit_candidate(x_train, y_train, x_val, y_val, layers, alpha):
    model = MLPClassifier(
        hidden_layer_sizes=layers, alpha=alpha,
        batch_size=64, learning_rate_init=0.001, random_state=SEED,
    )
    history = []
    best_loss, best_epoch, best_model = float("inf"), 0, None
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.partial_fit(x_train, y_train, classes=np.array([0, 1]))
        train_loss = float(log_loss(y_train, model.predict_proba(x_train)[:, 1]))
        val_loss = float(log_loss(y_val, model.predict_proba(x_val)[:, 1]))
        if not math.isfinite(val_loss) or not math.isfinite(train_loss):
            raise ValueError("Training produced non-finite loss")
        history.append({"epoch": epoch, "train_log_loss": train_loss, "validation_log_loss": val_loss})
        if val_loss < best_loss - MIN_DELTA:
            best_loss, best_epoch = val_loss, epoch
            best_model = copy.deepcopy(model)
        if epoch - best_epoch >= PATIENCE:
            break
    summary = {
        "layers": list(layers), "alpha": alpha, "best_epoch": best_epoch,
        "epochs_run": epoch, "validation_log_loss": best_loss,
        "seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary), flush=True)
    return best_model, summary, history


def main():
    started = time.perf_counter()
    raw_path = ROOT / "data/titanic.csv"
    raw = pd.read_csv(raw_path)
    if raw["survived"].isna().any() or not raw["survived"].isin([0, 1]).all():
        raise ValueError("Every passenger needs a binary survival label")
    if raw.duplicated().any():
        raise ValueError("Remove duplicate records before splitting")
    y = raw["survived"].to_numpy(dtype=int)
    indices = make_splits(raw, y)
    features = build_features(raw)
    preprocessor = make_preprocessor()
    x_train = preprocessor.fit_transform(features.iloc[indices["train"]])
    x_val = preprocessor.transform(features.iloc[indices["validation"]])
    y_train, y_val = y[indices["train"]], y[indices["validation"]]
    assert np.isfinite(x_train).all() and np.isfinite(x_val).all()
    models, summaries, histories = [], [], []
    for layers, alpha in CANDIDATES:
        model, summary, history = fit_candidate(x_train, y_train, x_val, y_val, layers, alpha)
        models.append(model)
        summaries.append(summary)
        histories.append(history)
    selected_index = int(np.argmin([item["validation_log_loss"] for item in summaries]))
    selected = models[selected_index]
    # A network without hidden layers is logistic regression; fixed training budget.
    baseline = MLPClassifier(hidden_layer_sizes=(), alpha=1.0, random_state=SEED)
    for _ in range(600):
        baseline.partial_fit(x_train, y_train)

    # Only now evaluate the chosen model and fixed baselines on test data.
    x_test = preprocessor.transform(features.iloc[indices["test"]])
    y_test = y[indices["test"]]
    probability = selected.predict_proba(x_test)[:, 1]
    model_metrics = evaluate(y_test, probability)
    pipeline = SurvivalPipeline(preprocessor, selected)
    model_dir, report_dir = ROOT / "models", ROOT / "reports"
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    bundle = {
        "feature_version": FEATURE_VERSION, "threshold": 0.5,
        "dataset_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    model_path = model_dir / "titanic_mlp.npz"
    save_model(model_path, pipeline, bundle)
    restored = load_model(model_path)[0].predict_proba(features.iloc[indices["test"]])[:, 1]
    np.testing.assert_allclose(restored, probability, rtol=0, atol=1e-14)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": bundle["dataset_sha256"],
        "seed": SEED,
        "split_method": "Greedy class-balanced ticket groups: 5 folds; folds 0–2 train, 3 validation, 4 test",
        "splits": {name: {"rows": len(index), "survived": int(y[index].sum())} for name, index in indices.items()},
        "input_features": features.columns.tolist(),
        "encoded_features": preprocessor.get_feature_names_out().tolist(),
        "excluded_columns": ["survived", "boat", "body", "name", "ticket", "home.dest"],
        "candidates": summaries, "selected_candidate": selected_index,
        "selection_metric": "validation_log_loss", "threshold": 0.5,
        "training": evaluate(y_train, selected.predict_proba(x_train)[:, 1]),
        "validation": evaluate(y_val, selected.predict_proba(x_val)[:, 1]),
        "test": model_metrics,
        "baselines_test": {
            "constant_train_prevalence": evaluate(y_test, np.full(len(y_test), y_train.mean())),
            "logistic_regression": evaluate(y_test, baseline.predict_proba(x_test)[:, 1]),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "total_seconds": time.perf_counter() - started,
    }
    save_json(report_dir / "metrics.json", report)
    save_json(report_dir / "split_indices.json", {name: index.tolist() for name, index in indices.items()})
    save_json(report_dir / "training_history.json", histories)
    pd.DataFrame({"source_row_index": indices["test"], "survived": y_test,
                  "survival_probability": probability, "predicted_survived": (probability >= 0.5).astype(int)}
                 ).to_csv(report_dir / "test_predictions.csv", index=False)
    write_report(report, report_dir / "training_report.md")
    print(json.dumps({"test": model_metrics, "total_seconds": report["total_seconds"]}, indent=2), flush=True)


def write_report(report, path):
    chosen = report["candidates"][report["selected_candidate"]]
    score = report["test"]
    comparison = [("Нейросеть MLP", score),
                  ("Постоянный прогноз", report["baselines_test"]["constant_train_prevalence"]),
                  ("Логистическая регрессия", report["baselines_test"]["logistic_regression"])]
    rows = "\n".join(f"| {name} | {metric['accuracy']:.1%} | {metric['roc_auc']:.3f} | {metric['f1']:.3f} | {metric['log_loss']:.3f} |"
                     for name, metric in comparison)
    splits = ", ".join(f"{name}: {item['rows']}" for name, item in report["splits"].items())
    tn, fp = score["confusion_matrix"][0]
    fn, tp = score["confusion_matrix"][1]
    text = f"""# Обучение Titanic

## Результат

Полносвязная нейросеть MLP, скрытые слои {chosen['layers']}, ReLU, выход sigmoid, Adam.
Число входов после преобразования: {len(report['encoded_features'])}.
Выбрана эпоха {chosen['best_epoch']}, L2 alpha = {chosen['alpha']}.
Порог прогноза: 0.5. Выбор архитектуры и эпохи выполнен по минимальной log loss на validation.

| Модель | Accuracy | ROC AUC | F1 выживших | Log loss |
| --- | ---: | ---: | ---: | ---: |
{rows}

Размер теста: {score['n']}. Матрица ошибок (строки — факт, столбцы — прогноз):

| | Прогноз: погиб | Прогноз: выжил |
| --- | ---: | ---: |
| Факт: погиб | {tn} | {fp} |
| Факт: выжил | {fn} | {tp} |

Precision выживших: {score['precision']:.1%}; recall: {score['recall']:.1%}.
Brier score: {score['brier_score']:.4f} (меньше — лучше).
Ориентировочный 95% интервал Wilson для accuracy: {score['accuracy_wilson_95'][0]:.1%}–{score['accuracy_wilson_95'][1]:.1%}.
Интервал рассчитан как для независимых пассажиров и может недооценивать неопределённость из-за связей внутри семей и билетов.

## Методика

Разбиение: {splits}. Пять приблизительно сбалансированных по исходу частей с группировкой по номеру билета.
Один билет не встречается одновременно в разных выборках. Seed = {SEED}.
Родственники с разными билетами всё же могут попадать в разные выборки.
Индексы строк (с нуля, без заголовка) сохранены в `split_indices.json`.

Медианы, индикаторы пропусков и масштабирование обучены только на train.
Категории заданы заранее по смыслу полей. Числовые входы: возраст, log(1 + стоимость),
число братьев/сестёр/супругов, число родителей/детей, размер семьи.
Категориальные входы: пол, класс билета, палуба, порт посадки.
Номер шлюпки и номер тела исключены как утечка ответа. Имя, номер билета и адрес не подаются в сеть.
Номер билета используется только для группировки при разбиении.

Проверены {len(report['candidates'])} заранее заданных варианта сети. Максимум {MAX_EPOCHS} эпох,
остановка после {PATIENCE} эпох без улучшения validation log loss минимум на {MIN_DELTA}.
Сохранены веса лучшей эпохи. Тест использован после выбора модели; по его результатам параметры не менялись.
Сохранённая модель обучалась только на train и именно она оценена на тесте.
Вероятности не проходили отдельную калибровку и являются оценкой модели, а не достоверным личным шансом.

## Время и воспроизводимость

Суммарное время обучения кандидатов: {sum(item['seconds'] for item in report['candidates']):.2f} с.
Весь запуск, включая обработку и оценку: {report['total_seconds']:.2f} с.
SHA-256 исходного CSV: `{report['dataset_sha256']}`.
Версии библиотек, метрики и параметры: `metrics.json`. История эпох: `training_history.json`.
Повторная загрузка модели проверена: вероятности совпадают с исходными с допуском 1e-14.

На 1 309 строках результаты чувствительны к разбиению. Большая сеть может переобучиться.
Для пользовательского интерфейса модель возвращает вероятность и бинарный прогноз.
Класс билета отражает условия проезда; точного места человека при крушении и статуса экипажа в данных нет.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
