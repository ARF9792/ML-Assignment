from dataclasses import dataclass
from math import ceil, comb
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


DATA_DIR = "Datasets"
ROLL_NO = "BT2024206"
N_SPLITS = 5
CV_RANDOM_STATE = 42
ROBUSTNESS_SEEDS = (0, 1, 7, 21, 42)

# The ordinary least-squares sweep uses every identifiable degree. Regularized
# models are then evaluated near that region, including degrees just beyond the
# OLS optimum. These grids were fixed before final model fitting.
REGULARIZATION_CONFIG = {
    1: {
        "max_degree": 10,
        "degrees": (3, 4, 5, 6),
        "ridge_alphas": (0.01, 0.1, 1.0, 10.0, 100.0),
        "lasso_alphas": (0.003, 0.01, 0.03, 0.1),
    },
    2: {
        "max_degree": 20,
        "degrees": (7, 8, 9, 10, 11, 12),
        "ridge_alphas": (0.01, 0.1, 1.0, 10.0, 100.0),
        "lasso_alphas": (0.001, 0.003, 0.01),
    },
}


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    degree: int
    alpha: float | None
    train_mse: float
    validation_mse: float
    validation_r2: float

    @property
    def label(self):
        if self.alpha is None:
            return f"{self.model_name}, degree {self.degree}"
        return f"{self.model_name}, degree {self.degree}, alpha={self.alpha:g}"


def load_data(var_id):
    train_path = os.path.join(DATA_DIR, f"{ROLL_NO}_train_var{var_id}.csv")
    test_path = os.path.join(DATA_DIR, f"{ROLL_NO}_test_var{var_id}.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["y"]).values
    y_train = train_df["y"].values
    X_test = test_df.drop(columns=["y"], errors="ignore").values
    return X_train, y_train, X_test


def make_regressor(model_name, alpha=None):
    if model_name == "OLS":
        return LinearRegression()
    if model_name == "Ridge":
        return Ridge(alpha=alpha)
    if model_name == "Lasso":
        return Lasso(alpha=alpha, max_iter=30_000, tol=1e-4)
    raise ValueError(f"Unknown model: {model_name}")


def make_pipeline(degree, model_name, alpha=None):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", make_regressor(model_name, alpha)),
    ])


def evaluate_candidate(X, y, degree, model_name, alpha, cv):
    pipeline = make_pipeline(degree, model_name, alpha)
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        return_train_score=True,
        n_jobs=1,
    )
    return ModelResult(
        model_name=model_name,
        degree=degree,
        alpha=alpha,
        train_mse=-scores["train_mse"].mean(),
        validation_mse=-scores["test_mse"].mean(),
        validation_r2=scores["test_r2"].mean(),
    )


def identifiable_ols_degrees(X, max_degree):
    min_fold_train_size = len(X) - ceil(len(X) / N_SPLITS)
    degrees = []
    for degree in range(1, max_degree + 1):
        # C(p+d, d) includes every polynomial coefficient and the intercept.
        n_parameters = comb(X.shape[1] + degree, degree)
        if n_parameters >= min_fold_train_size:
            print(
                f"OLS stopping rule: degree {degree} needs {n_parameters} "
                f"parameters, while a CV training fold has "
                f"{min_fold_train_size} samples."
            )
            break
        degrees.append(degree)
    return degrees


def select_model(X, y, var_id):
    config = REGULARIZATION_CONFIG[var_id]
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    results = []

    ols_degrees = identifiable_ols_degrees(X, config["max_degree"])
    print(f"Phase {var_id}: evaluating OLS degrees {ols_degrees[0]}-{ols_degrees[-1]}.")
    for degree in ols_degrees:
        results.append(evaluate_candidate(X, y, degree, "OLS", None, cv))

    best_ols = min(
        (result for result in results if result.model_name == "OLS"),
        key=lambda result: result.validation_mse,
    )
    print(f"Phase {var_id}: best unregularized model: {best_ols.label}.")

    print(
        f"Phase {var_id}: evaluating regularized degrees "
        f"{config['degrees'][0]}-{config['degrees'][-1]}."
    )
    for degree in config["degrees"]:
        for alpha in config["ridge_alphas"]:
            results.append(evaluate_candidate(X, y, degree, "Ridge", alpha, cv))
        for alpha in config["lasso_alphas"]:
            results.append(evaluate_candidate(X, y, degree, "Lasso", alpha, cv))

    best_overall = min(results, key=lambda result: result.validation_mse)
    best_by_family = {
        family: min(
            (result for result in results if result.model_name == family),
            key=lambda result: result.validation_mse,
        )
        for family in ("OLS", "Ridge", "Lasso")
    }

    print(f"Phase {var_id}: model comparison")
    for family in ("OLS", "Ridge", "Lasso"):
        result = best_by_family[family]
        print(
            f"  {result.label}: validation MSE={result.validation_mse:.6f}, "
            f"validation R2={result.validation_r2:.6f}"
        )
    print(f"Phase {var_id}: selected {best_overall.label}.\n")
    return best_overall, best_ols, results


def plot_model_comparison(results, selected, var_id):
    plt.figure(figsize=(10, 6))
    colors = {"OLS": "tab:blue", "Ridge": "tab:green", "Lasso": "tab:orange"}

    for family in ("OLS", "Ridge", "Lasso"):
        family_results = [result for result in results if result.model_name == family]
        degrees = sorted({result.degree for result in family_results})
        best_at_degree = [
            min(
                (result for result in family_results if result.degree == degree),
                key=lambda result: result.validation_mse,
            )
            for degree in degrees
        ]
        plt.plot(
            degrees,
            [result.validation_mse for result in best_at_degree],
            label=(
                "OLS"
                if family == "OLS"
                else f"{family} (best alpha at each degree)"
            ),
            color=colors[family],
            marker="o",
        )

    plt.scatter(
        [selected.degree],
        [selected.validation_mse],
        marker="*",
        s=230,
        color="crimson",
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
        label=f"Selected: {selected.label}",
    )
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Cross-Validation MSE")
    plt.title(f"Polynomial Model Selection for Phase {var_id}")
    plt.yscale("log")
    plt.xticks(sorted({result.degree for result in results}))
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"var{var_id}_learning_curve.png", dpi=150)
    plt.close()


def repeated_cv_mse(X, y, result):
    seed_mses = []
    for seed in ROBUSTNESS_SEEDS:
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        scores = cross_validate(
            make_pipeline(result.degree, result.model_name, result.alpha),
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
        seed_mses.append(-scores["test_score"].mean())
    return float(np.mean(seed_mses)), float(np.std(seed_mses, ddof=1))


def train_and_predict(X_train, y_train, X_test, selected, var_id):
    pipeline = make_pipeline(selected.degree, selected.model_name, selected.alpha)
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"Phase {var_id} final model: {selected.label}")
    print(f"Phase {var_id} final training MSE: {train_mse:.6f}")
    print(f"Phase {var_id} final training R2: {train_r2:.6f}")
    return pipeline.predict(X_test)


def process_phase(var_id):
    print(f"\nProcessing Phase {var_id} (var{var_id})...")
    X_train, y_train, X_test = load_data(var_id)
    selected, best_ols, results = select_model(X_train, y_train, var_id)

    ols_repeat_mean, ols_repeat_sd = repeated_cv_mse(X_train, y_train, best_ols)
    selected_repeat_mean, selected_repeat_sd = repeated_cv_mse(X_train, y_train, selected)
    print(
        f"Phase {var_id} repeated-CV MSE: OLS={ols_repeat_mean:.6f} "
        f"(SD {ols_repeat_sd:.6f}), selected={selected_repeat_mean:.6f} "
        f"(SD {selected_repeat_sd:.6f})"
    )

    plot_model_comparison(results, selected, var_id)
    predictions = train_and_predict(X_train, y_train, X_test, selected, var_id)
    output_path = f"{ROLL_NO}_pred_var{var_id}.csv"
    pd.DataFrame({"y": predictions}).to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def main():
    print("Running regularized polynomial regression assignment")
    process_phase(1)
    process_phase(2)
    print("\nProcess complete: predictions and model-selection plots were regenerated.")


if __name__ == "__main__":
    main()
