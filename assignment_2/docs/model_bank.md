# Assignment 2 model bank (Step 6)

Run from the workspace root:

```bash
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_train.py --save-model
```

The command repeats the Step 5 development process: compare Logistic Regression and XGBoost fitted on 2023 Train, select XGBoost using January–April 2024 Validation ROC-AUC (0.8409), then fit a **fresh complete XGBoost Pipeline** on Train + Validation. The final fit uses 7,985 applications, 2,287 defaults, 71 ordered input fields, and the 2024-10-31 label-knowledge cut-off. It retains the selected XGBoost configuration without further tuning. Numeric median imputation, categorical most-frequent imputation, one-hot encoding, and the fitted model are stored together.

```text
model_bank/
└── xgboost_20241031_v1/
    ├── model.joblib
    └── metadata.json
```

`metadata.json` records the version, target (`30dpd_6mob`), fit date and application period, training counts, exact input and transformed feature order, numeric/categorical lists, selected model parameters, Validation selection AUC, final OOT metrics, and reserved inference months. The saved champion's independent retrospective May–October 2024 OOT check covers 2,986 applications and 897 defaults: **ROC-AUC 0.8203; Gini 0.6406**. OOT did not influence selection or fitting.

Later code can call `load_model(model_bank_directory, "xgboost_20241031_v1")` from `utils.model_store` to receive `(fitted_pipeline, metadata)`. Loading only deserializes files; it performs no fitting or preprocessing recalculation. In the Step 6 validation, the complete saved pipeline produced identical probabilities on a fixed, `loan_id`-sorted 20-row OOT sample: maximum absolute difference from the in-memory pipeline was **0.0**. A separate process also loaded the artefact and reproduced the sample probability sum without training.

The November 2024–January 2025 cohort remains reserved for later monthly inference and monitoring. No monthly predictions or monitoring tables are created by this command.
