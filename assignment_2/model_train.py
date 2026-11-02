"""Compare models, refit the selected one, and optionally save it."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from utils.modelling_data import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    load_modelling_dataset,
)
from utils.model_store import load_model, save_model


FIT_DATE = "2024-10-31"
OOT_EVALUATION_DATE = "2025-04-30"
GOLD_DIRECTORY = Path(__file__).resolve().parent / "datamart" / "gold"
MODEL_BANK = Path(__file__).resolve().parent / "model_bank"
MODEL_VERSION = "xgboost_20241031_v1"


def make_pipeline(name: str) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if name == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))
        estimator = LogisticRegression(max_iter=2000, random_state=42)
    elif name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=5.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=2,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown candidate: {name}")

    preprocessing = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def metrics(pipeline: Pipeline, frame) -> dict:
    probabilities = pipeline.predict_proba(frame.loc[:, MODEL_FEATURES])[:, 1]
    auc = roc_auc_score(frame["label"], probabilities)
    return {
        "rows": len(frame),
        "defaults": int(frame["label"].sum()),
        "default_rate": float(frame["label"].mean()),
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
    }


def top_features(pipeline: Pipeline, name: str) -> list[dict]:
    transformed_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    if name == "logistic_regression":
        values = pipeline.named_steps["model"].coef_[0]
        order = np.argsort(np.abs(values))[::-1][:10]
        return [
            {"feature": str(transformed_names[i]), "coefficient": float(values[i])}
            for i in order
        ]
    values = pipeline.named_steps["model"].feature_importances_
    order = np.argsort(values)[::-1][:10]
    return [
        {"feature": str(transformed_names[i]), "importance": float(values[i])}
        for i in order
    ]


def check_preprocessing_fit_on(pipeline: Pipeline, fitting_rows) -> None:
    """Check learned preprocessing statistics against the rows used for fitting."""
    preprocessing = pipeline.named_steps["preprocessing"]
    numeric = preprocessing.named_transformers_["numeric"]
    numeric_imputer = numeric.named_steps["imputer"]
    fitting_numeric = fitting_rows.loc[:, NUMERIC_FEATURES]
    expected_medians = fitting_numeric.median().to_numpy(dtype=float)
    np.testing.assert_allclose(numeric_imputer.statistics_, expected_medians)

    categorical = preprocessing.named_transformers_["categorical"]
    categorical_imputer = categorical.named_steps["imputer"]
    encoder = categorical.named_steps["onehot"]
    for i, feature in enumerate(CATEGORICAL_FEATURES):
        assert categorical_imputer.statistics_[i] in fitting_rows[feature].mode(dropna=True).tolist()
        fitting_values = fitting_rows[feature].fillna(categorical_imputer.statistics_[i])
        assert set(encoder.categories_[i]) == set(fitting_values.unique())

    if "scaler" in numeric.named_steps:
        expected_scaled_input = numeric_imputer.transform(fitting_numeric)
        np.testing.assert_allclose(
            numeric.named_steps["scaler"].mean_, expected_scaled_input.mean(axis=0)
        )


def main(save_champion: bool = False) -> None:
    spark = (
        SparkSession.builder.appName("assignment2-model-development")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        mature = load_modelling_dataset(spark, GOLD_DIRECTORY, FIT_DATE)
        train = mature.filter(
            (F.col("feature_snapshot_date") >= "2023-01-01")
            & (F.col("feature_snapshot_date") < "2024-01-01")
        ).toPandas()
        validation = mature.filter(
            (F.col("feature_snapshot_date") >= "2024-01-01")
            & (F.col("feature_snapshot_date") < "2024-05-01")
        ).toPandas()
        assert not train.empty and not validation.empty
        assert len(set(train.loan_id) & set(validation.loan_id)) == 0
        assert train.label.nunique() == validation.label.nunique() == 2
        assert max(train.label_snapshot_date.max(), validation.label_snapshot_date.max()).isoformat() <= FIT_DATE
        assert len(set(MODEL_FEATURES)) == len(MODEL_FEATURES)
        assert set(MODEL_FEATURES).isdisjoint(
            {"loan_id", "Customer_ID", "feature_snapshot_date", "label_snapshot_date", "label", "dpd", "overdue_amt", "paid_amt"}
        )

        candidates = {}
        fitted = {}
        for name in ("logistic_regression", "xgboost"):
            pipeline = make_pipeline(name)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.fit(train.loc[:, MODEL_FEATURES], train["label"])
            check_preprocessing_fit_on(pipeline, train)
            candidates[name] = {
                "train": metrics(pipeline, train),
                "validation": metrics(pipeline, validation),
                "transformed_features": len(
                    pipeline.named_steps["preprocessing"].get_feature_names_out()
                ),
                "top_features": top_features(pipeline, name),
                "fit_warnings": [str(w.message) for w in caught],
            }
            fitted[name] = pipeline

        # Validation alone determines the winner. A tie favors the simpler baseline.
        champion_name = max(
            ("logistic_regression", "xgboost"),
            key=lambda name: candidates[name]["validation"]["roc_auc"],
        )

        # Refit a fresh, identical pipeline on all mature development rows.
        final_training = pd.concat([train, validation], ignore_index=True)
        assert final_training.loan_id.is_unique
        final_champion = make_pipeline(champion_name)
        with warnings.catch_warnings(record=True) as refit_warnings:
            warnings.simplefilter("always")
            final_champion.fit(final_training.loc[:, MODEL_FEATURES], final_training["label"])
        check_preprocessing_fit_on(final_champion, final_training)
        train_only_medians = fitted[champion_name].named_steps["preprocessing"].named_transformers_["numeric"].named_steps["imputer"].statistics_
        final_medians = final_champion.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["imputer"].statistics_
        changed_medians = int(np.count_nonzero(~np.isclose(train_only_medians, final_medians)))

        # Load OOT only after selecting and refitting the final champion.
        oot = load_modelling_dataset(spark, GOLD_DIRECTORY, OOT_EVALUATION_DATE).filter(
            (F.col("feature_snapshot_date") >= "2024-05-01")
            & (F.col("feature_snapshot_date") < "2024-11-01")
        ).toPandas()
        assert not oot.empty and oot.label.nunique() == 2
        assert len(set(final_training.loan_id) & set(oot.loan_id)) == 0
        assert oot.label_snapshot_date.max().isoformat() <= OOT_EVALUATION_DATE

        final_oot = metrics(final_champion, oot)
        result = {
            "fit_date": FIT_DATE,
            "oot_evaluation_date": OOT_EVALUATION_DATE,
            "input_features": len(MODEL_FEATURES),
            "candidates": candidates,
            "champion": champion_name,
            "final_refit": {
                "rows": len(final_training),
                "defaults": int(final_training["label"].sum()),
                "application_start": str(final_training.feature_snapshot_date.min()),
                "application_end": str(final_training.feature_snapshot_date.max()),
                "transformed_features": len(final_champion.named_steps["preprocessing"].get_feature_names_out()),
                "numeric_medians_changed_from_train_only": changed_medians,
                "top_features": top_features(final_champion, champion_name),
                "fit_warnings": [str(w.message) for w in refit_warnings],
            },
            "final_champion_oot": final_oot,
            "reserved_2024_11_to_2025_01_used": False,
        }

        if save_champion:
            transformed_names = [
                str(name) for name in final_champion.named_steps["preprocessing"].get_feature_names_out()
            ]
            model_type = {
                "xgboost": "XGBoost",
                "logistic_regression": "Logistic Regression",
            }[champion_name]
            parameter_names = {
                "xgboost": (
                    "n_estimators", "max_depth", "learning_rate", "subsample",
                    "colsample_bytree", "reg_lambda", "objective", "eval_metric",
                    "tree_method", "random_state", "n_jobs",
                ),
                "logistic_regression": ("C", "penalty", "solver", "max_iter", "random_state"),
            }[champion_name]
            selected_parameters = final_champion.named_steps["model"].get_params()
            metadata = {
                "model_version": MODEL_VERSION,
                "model_type": model_type,
                "target_definition": "30dpd_6mob",
                "fit_date": FIT_DATE,
                "training_application_start": "2023-01-01",
                "training_application_end": str(final_training.feature_snapshot_date.max()),
                "training_rows": len(final_training),
                "training_defaults": int(final_training["label"].sum()),
                "input_feature_count": len(MODEL_FEATURES),
                "input_features": list(MODEL_FEATURES),
                "numeric_features": list(NUMERIC_FEATURES),
                "categorical_features": list(CATEGORICAL_FEATURES),
                "transformed_feature_count": len(transformed_names),
                "transformed_feature_names": transformed_names,
                "model_parameters": {name: selected_parameters[name] for name in parameter_names},
                "candidate_selection_metric": "Validation ROC-AUC",
                "candidate_validation_auc": candidates[champion_name]["validation"]["roc_auc"],
                "final_oot_auc": final_oot["roc_auc"],
                "final_oot_gini": final_oot["gini"],
                "final_oot_period": "2024-05 through 2024-10",
                "final_oot_rows": final_oot["rows"],
                "final_oot_defaults": final_oot["defaults"],
                "champion_selection_used_oot": False,
                "reserved_inference_period": "2024-11 through 2025-01",
            }
            model_directory = save_model(final_champion, metadata, MODEL_BANK)
            reloaded, loaded_metadata = load_model(MODEL_BANK, MODEL_VERSION)
            assert loaded_metadata == metadata
            assert isinstance(reloaded, Pipeline) and hasattr(reloaded, "predict_proba")
            assert loaded_metadata["model_version"] == model_directory.name
            assert loaded_metadata["input_feature_count"] == len(MODEL_FEATURES)
            assert loaded_metadata["input_features"] == list(MODEL_FEATURES)
            assert loaded_metadata["numeric_features"] == list(NUMERIC_FEATURES)
            assert loaded_metadata["categorical_features"] == list(CATEGORICAL_FEATURES)
            loaded_names = list(reloaded.named_steps["preprocessing"].get_feature_names_out())
            assert loaded_names == transformed_names == loaded_metadata["transformed_feature_names"]
            assert len(loaded_names) == loaded_metadata["transformed_feature_count"]
            assert list(reloaded.named_steps["preprocessing"].feature_names_in_) == list(MODEL_FEATURES)
            assert set(MODEL_FEATURES).isdisjoint(
                {"loan_id", "Customer_ID", "feature_snapshot_date", "label_snapshot_date", "label", "dpd", "overdue_amt", "paid_amt"}
            )
            sample = oot.sort_values("loan_id").head(20).loc[:, MODEL_FEATURES]
            original_probabilities = final_champion.predict_proba(sample)[:, 1]
            loaded_probabilities = reloaded.predict_proba(sample)[:, 1]
            np.testing.assert_allclose(original_probabilities, loaded_probabilities, rtol=0, atol=1e-12)
            result["model_bank"] = {
                "model_version": MODEL_VERSION,
                "path": str(model_directory),
                "reload_sample_rows": len(sample),
                "reload_sample_probability_sum": float(original_probabilities.sum()),
                "max_absolute_probability_difference": float(
                    np.max(np.abs(original_probabilities - loaded_probabilities))
                ),
            }
        print(
            "STEP5_RESULT="
            + json.dumps(result)
        )
    finally:
        spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the Assignment 2 champion")
    parser.add_argument("--save-model", action="store_true", help="Save and reload the final champion")
    main(save_champion=parser.parse_args().save_model)
