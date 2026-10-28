# Assignment 2 monthly inference (Step 7)

The frozen `xgboost_20241031_v1` champion is replayed month by month on later loan applications. This is a retrospective production-like simulation, not a claim that the model was historically deployed. Inference reads only `datamart/gold/feature_store/` and the saved model bank artefact. It does not read `label_store`, call `fit`, or recalculate preprocessing. MOB 6 outcomes become known later and are outside this step.

From the workspace root, run one month at a time:

```bash
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_inference.py --snapshot-date 2024-11-01
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_inference.py --snapshot-date 2024-12-01
docker compose -f assignment_2/docker-compose.yaml run --rm --no-deps jupyter python model_inference.py --snapshot-date 2025-01-01
```

The script loads the complete fitted Pipeline with `utils.model_store.load_model`, checks the 71 required fields and their numeric/categorical types, and passes them to `predict_proba` in the exact order recorded in `metadata.json`. It requires an application month after the 2024-10-31 model fitting cut-off. Its only prediction source is the selected month's application-time Gold feature rows.

The Gold table is `datamart/gold/model_predictions/`, partitioned by `feature_snapshot_date`. Its columns are `loan_id` (string), `Customer_ID` (string), `feature_snapshot_date` (date), `model_version` (string), and `predicted_default_probability` (double). No label, DPD, overdue amount, paid amount, or future repayment field is stored. A rerun overwrites only the requested month's partition.

| Application month | Predictions | Unique loans | Min probability | Max probability | Mean probability | Median probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-11-01 | 488 | 488 | 0.021580 | 0.731027 | 0.277598 | 0.163750 |
| 2024-12-01 | 515 | 515 | 0.016178 | 0.732174 | 0.242681 | 0.152694 |
| 2025-01-01 | 526 | 526 | 0.022250 | 0.735901 | 0.278643 | 0.166584 |

Combined validation: **1,529 predictions for 1,529 distinct loans**; exact one-to-one coverage of the eligible feature rows; zero duplicate `(loan_id, model_version)` keys; no null IDs or probabilities; all scores in `[0, 1]`; one model version and exactly the three requested monthly partitions. Replaying November still produced 488 distinct rows, and the December/January partition files remained byte-for-byte unchanged. The table reports score distributions, not model performance; no AUC, Gini, PSI, or monitoring table is produced here.
