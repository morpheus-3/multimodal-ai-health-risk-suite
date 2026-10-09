"""Leakage-safe tabular + lifestyle ensemble for two research risk targets."""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from imblearn.over_sampling import SMOTE
from .common import directory, metadata

FEATURES = ["age", "bmi", "systolic_bp", "glucose", "cholesterol", "smoker", "activity_minutes"]
TARGETS = ["cardiovascular", "diabetes"]
LIMITS = [(18, 100), (10, 70), (70, 250), (40, 500), (80, 500), (0, 1), (0, 300)]

def validate(values):
    row = np.asarray([values[k] for k in FEATURES], dtype=float)
    if not np.isfinite(row).all() or any(not lo <= v <= hi for v, (lo, hi) in zip(row, LIMITS)) or row[5] not in (0, 1):
        raise ValueError("Health features are outside supported ranges.")
    return row.reshape(1, -1)

def train(csv, epochs=12, synthetic=False):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    frame = pd.read_csv(csv)
    for row in frame[FEATURES].to_dict("records"):
        validate(row)
    if len(frame) < 60 or frame[TARGETS].isna().any().any():
        raise ValueError("Provide at least 60 complete labeled rows.")
    path = directory("health")
    report = {"synthetic": synthetic, "rows": len(frame), "targets": {}, "features": FEATURES}
    for target in TARGETS:
        if set(frame[target].unique()) != {0, 1} or frame[target].value_counts().min() < 10:
            raise ValueError("Each health target needs at least ten rows per binary class.")
        xtrain, xtest, ytrain, ytest = train_test_split(frame[FEATURES].to_numpy(), frame[target].to_numpy(), test_size=.25, stratify=frame[target], random_state=42)
        scaler = StandardScaler().fit(xtrain)
        xs = scaler.transform(xtrain)
        xs, ys = SMOTE(random_state=42, k_neighbors=min(5, int(np.bincount(ytrain).min())-1)).fit_resample(xs, ytrain)
        forest = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42).fit(xs, ys)
        linear = LogisticRegression(max_iter=500).fit(xs, ys)
        neural = tf.keras.Sequential([tf.keras.Input((len(FEATURES),)), tf.keras.layers.Dense(24, activation="relu"), tf.keras.layers.Dense(12, activation="relu"), tf.keras.layers.Dense(1, activation="sigmoid")])
        neural.compile(optimizer="adam", loss="binary_crossentropy")
        neural.fit(xs, ys, epochs=epochs, batch_size=32, verbose=0)
        joblib.dump({"scaler": scaler, "forest": forest, "linear": linear, "background": xs[:12]}, path / f"{target}.joblib")
        neural.save(path / f"{target}.keras")
        test = scaler.transform(xtest)
        probability = (forest.predict_proba(test)[:, 1] + linear.predict_proba(test)[:, 1] + neural(test, training=False).numpy().ravel()) / 3
        report["targets"][target] = {"roc_auc": float(roc_auc_score(ytest, probability)), "brier": float(brier_score_loss(ytest, probability)), "held_out_rows": len(ytest)}
    metadata(path / "report.json", report)
    return report

def predict(values):
    import tensorflow as tf
    import shap
    row = validate(values)
    path = directory("health")
    report = metadata(path / "report.json")
    results = {}
    for target in TARGETS:
        saved = joblib.load(path / f"{target}.joblib")
        neural = tf.keras.models.load_model(path / f"{target}.keras", compile=False)
        x = saved["scaler"].transform(row)
        def ensemble(z):
            return (saved["forest"].predict_proba(z)[:, 1] + saved["linear"].predict_proba(z)[:, 1] + neural(z, training=False).numpy().ravel()) / 3
        explanation = shap.KernelExplainer(ensemble, saved["background"])
        contributions = np.asarray(explanation.shap_values(x, nsamples=80, silent=True)).reshape(-1)
        probability = float(ensemble(x)[0])
        results[target] = {"probability": probability, "band": "high" if probability >= .7 else "moderate" if probability >= .4 else "low", "base_probability": float(explanation.expected_value), "shap": dict(zip(FEATURES, map(float, contributions)))}
    return {"results": results, "evaluation": report, "notice": "Research demonstration; not a diagnosis or clinically validated risk calculator. SHAP explains model behavior, not causation."}
