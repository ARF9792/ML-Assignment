# Polynomial Regression Assignment

Polynomial regression models for two personalized geothermal-engineering datasets assigned to roll number **BT2024206**.

Repository: [github.com/ARF9792/ML-Assignment](https://github.com/ARF9792/ML-Assignment)

## Final Models

| Problem | Selected model | 5-fold CV MSE | 5-fold CV R2 |
|---|---|---:|---:|
| var1 | Degree-5 polynomial + Lasso (`alpha=0.01`) | 0.362188 | 0.966917 |
| var2 | Degree-10 polynomial + Ridge (`alpha=1`) | 0.255827 | 0.994409 |

The models were selected using shuffled 5-fold cross-validation. A repeated check across five different fold partitions confirmed that both regularized models outperform their best unregularized alternatives.

## Repository Contents

- `assignment_solution.py`: model selection, repeated validation, final training, plotting, and test prediction
- `Datasets/`: personalized train and test CSV files
- `BT2024206_pred_var1.csv`: final predictions for var1
- `BT2024206_pred_var2.csv`: final predictions for var2
- `Report.md`: report source
- `output/pdf/BT2024206_Polynomial_Regression_Report.pdf`: final report
- `var1_learning_curve.png` and `var2_learning_curve.png`: model-selection plots

## Running the Code

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python assignment_solution.py
```

The script expects the four personalized datasets inside `Datasets/`. It regenerates both prediction CSVs and both model-selection plots.

## Method Summary

Each model uses total-degree polynomial expansion followed by feature standardization. Ordinary least squares, Ridge, and Lasso candidates are compared using validation MSE. All preprocessing is performed inside a Scikit-Learn pipeline to avoid cross-validation leakage.
