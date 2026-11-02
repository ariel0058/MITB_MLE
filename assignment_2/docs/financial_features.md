# Financial features

I added four fields (referenced from sample assignments) to the Gold feature store using the cleaned Silver financial record matched on customer and application date. The clickstream calculations are unchanged.

| Gold field | Formula | Interpretation | Null and zero handling |
| --- | --- | --- | --- |
| `debt_to_salary` | `Outstanding_Debt / Monthly_Inhand_Salary` | Outstanding debt relative to monthly take-home salary. | Null if debt or salary is missing, or salary is `<= 0`. Zero debt gives zero when salary is positive. |
| `emi_to_salary` | `Total_EMI_per_month / Monthly_Inhand_Salary` | Existing monthly EMI relative to monthly take-home salary. | Null if EMI or salary is missing, or salary is `<= 0`. Zero EMI gives zero when salary is positive. |
| `repayment_ability` | `Monthly_Inhand_Salary - Total_EMI_per_month` | Income remaining after existing monthly EMI, as an approximation. | Null if either input is missing. Negative values are retained. |
| `num_fin_products` | `Num_Bank_Accounts + Num_Credit_Card + Num_of_Loan` | Sum of three product counts; it is not a distinct-product count. | Null if any count is missing; missing counts are not treated as zero. |

## Implementation location

`build_feature_store` in `utils/data_processing_gold_table.py` calculates the fields without changing the Silver source columns or the one-row-per-loan Gold grain. 

All inputs come from the financial snapshot where `(Customer_ID, snapshot_date)` matches the loan's `(Customer_ID, feature_snapshot_date)`. 
