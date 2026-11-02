# Model bank

Run from the workspace root:

```bash
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_train.py --save-model
```

This command repeats the model selection process and refits the selected XGBoost model on Train + Validation. XGBoost was selected based on the January–April 2024 Validation AUC of 0.8409.

The final model is fitted on 7,985 applications with 2,287 defaults, using 71 input features and labels available by 2024-10-31. The full fitted pipeline is saved, including preprocessing and XGBoost, so the same imputation and one-hot encoding can be reused during inference.

```text
model_bank/
└── xgboost_20241031_v1/
    ├── model.joblib
    └── metadata.json
```

model.joblib stores the fitted pipeline. metadata.json stores the model version, target (30dpd_6mob), training period and counts, input and transformed feature lists, model parameters, Validation AUC, OOT results, and the reserved inference period.

The final model was also checked on the May–October 2024 retrospective OOT set, which contains 2,986 applications and 897 defaults. It achieved ROC-AUC 0.8203 and Gini 0.6406. OOT was evaluated only after model selection and refitting.

The saved model can be loaded using:
```load_model(model_bank_directory, "xgboost_20241031_v1")```
This returns the fitted pipeline and its metadata without retraining or refitting preprocessing.

To check that saving and loading did not change the model, I compared predictions on a fixed 20-row OOT sample sorted by loan_id. The probabilities before and after reloading were identical, with a maximum absolute difference of 0.0.

Applications from November 2024 to January 2025 are not used for model fitting. They are reserved for the later inference and monitoring steps. This command only creates the model bank artefact.