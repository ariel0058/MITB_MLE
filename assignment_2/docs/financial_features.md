# Assignment 2 financial features (implemented in Step 3)

These four fields extend the existing application-time Gold feature store. All inputs are the cleaned numeric columns in the customer's Silver financials row where `(Customer_ID, snapshot_date)` matches `(Customer_ID, feature_snapshot_date)` for the loan application. The existing join and clickstream logic remain unchanged.

| Gold field | Formula | Business meaning | Source columns | Null / zero handling | Point-in-time rule |
| --- | --- | --- | --- | --- | --- |
| `debt_to_salary` | `Outstanding_Debt / Monthly_Inhand_Salary` | Outstanding debt expressed relative to monthly take-home salary. | `Outstanding_Debt`, `Monthly_Inhand_Salary` | Null if debt or salary is null, or salary is `<= 0`; otherwise divide directly. Do not add 1 to salary. Zero debt with positive salary gives 0. | Both values come from the financial snapshot on the loan application date. |
| `emi_to_salary` | `Total_EMI_per_month / Monthly_Inhand_Salary` | Existing monthly EMI commitments as a share of monthly take-home salary. | `Total_EMI_per_month`, `Monthly_Inhand_Salary` | Null if EMI or salary is null, or salary is `<= 0`; otherwise divide directly. Do not add 1 to salary. Zero EMI with positive salary gives 0. | Both values come from the financial snapshot on the loan application date. |
| `repayment_ability` | `Monthly_Inhand_Salary - Total_EMI_per_month` | Approximate monthly income remaining after existing EMI commitments. | `Monthly_Inhand_Salary`, `Total_EMI_per_month` | Null if either input is null. Retain negative results; zero is valid. | Both values come from the financial snapshot on the loan application date. |
| `num_fin_products` | `Num_Bank_Accounts + Num_Credit_Card + Num_of_Loan` | Simple count-based proxy for financial product exposure or complexity; it is not a distinct-product count. | `Num_Bank_Accounts`, `Num_Credit_Card`, `Num_of_Loan` | Null if any component is null. Do not replace missing counts with zero. All-zero inputs give 0. | All three counts come from the financial snapshot on the loan application date. |

## Implementation location

The four fields are calculated in `build_feature_store` in `utils/data_processing_gold_table.py` from the already cleaned Silver financials. The Silver source columns are unchanged. The Gold store preserves one row per `loan_id` and the existing application-date join.

Only application-time financial values may enter these formulas. They do not use subsequent repayments, overdue amounts, balances, DPD, or the MOB 6 label. The existing previous-three-complete-months clickstream logic is unaffected.
