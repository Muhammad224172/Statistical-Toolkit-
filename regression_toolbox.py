from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class RegressionToolbox:
    """Linear regression with explicit matrix calculations and hypothesis tests."""

    def __init__(self, data: str | Path | pd.DataFrame | None = None):
        self.data: pd.DataFrame | None = None
        if data is not None:
            self.load_data(data)

    def load_data(self, path_or_dataframe):
        if isinstance(path_or_dataframe, pd.DataFrame):
            self.data = path_or_dataframe.copy()
            return self.data
        path = Path(path_or_dataframe)
        if path.suffix.lower() == ".csv":
            self.data = pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            self.data = pd.read_excel(path)
        elif path.suffix.lower() == ".json":
            self.data = pd.read_json(path)
        else:
            raise ValueError("Supported data formats are CSV, Excel, JSON, or a pandas DataFrame.")
        return self.data

    def summarize_data(self, columns=None):
        df = self._df()
        if columns:
            df = df[columns]
        return df.describe(include="all").T

    def fit_simple_linear_regression(self, x_col, y_col, alpha=0.05):
        return self.fit_multiple_linear_regression([x_col], y_col, alpha)

    def fit_multiple_linear_regression(self, x_cols, y_col, alpha=0.05):
        alpha = self._check_alpha(alpha)
        cols = [y_col] + list(x_cols)
        df = self._df()[cols].apply(pd.to_numeric, errors="coerce").dropna()
        y = df[y_col].to_numpy(float)
        X_raw = df[x_cols].to_numpy(float)
        X = np.column_stack([np.ones(len(df)), X_raw])
        names = ["Intercept"] + list(x_cols)
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        fitted = X @ beta
        residuals = y - fitted
        n, p = X.shape
        df_resid = n - p
        sse = float(residuals @ residuals)
        mse = sse / df_resid
        cov_beta = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov_beta))
        t_stats = beta / se
        p_values = 2 * stats.t(df=df_resid).sf(np.abs(t_stats))
        tcrit = stats.t(df=df_resid).ppf(1 - alpha / 2)
        ci_low, ci_high = beta - tcrit * se, beta + tcrit * se
        sst = float(((y - y.mean()) ** 2).sum())
        ssr = sst - sse
        r2 = ssr / sst
        adj_r2 = 1 - (1 - r2) * (n - 1) / df_resid
        f_stat = (ssr / (p - 1)) / mse if p > 1 else np.nan
        f_p = stats.f.sf(f_stat, p - 1, df_resid) if p > 1 else np.nan
        return {
            "method": "linear regression",
            "x_cols": list(x_cols),
            "y_col": y_col,
            "data": df,
            "X": X,
            "y": y,
            "coefficient_names": names,
            "coefficients": beta,
            "cov_beta": cov_beta,
            "standard_errors": se,
            "t_statistics": t_stats,
            "p_values": p_values,
            "confidence_intervals": np.column_stack([ci_low, ci_high]),
            "fitted": fitted,
            "residuals": residuals,
            "n": n,
            "p": p,
            "df_model": p - 1,
            "df_resid": df_resid,
            "sse": sse,
            "ssr": ssr,
            "sst": sst,
            "mse": mse,
            "residual_standard_error": np.sqrt(mse),
            "r_squared": r2,
            "adjusted_r_squared": adj_r2,
            "overall_f": f_stat,
            "overall_f_p_value": f_p,
            "alpha": alpha,
        }

    def coefficient_table(self, result):
        return pd.DataFrame({
            "term": result["coefficient_names"],
            "estimate": result["coefficients"],
            "std_error": result["standard_errors"],
            "t": result["t_statistics"],
            "p_value": result["p_values"],
            "ci_low": result["confidence_intervals"][:, 0],
            "ci_high": result["confidence_intervals"][:, 1],
        })

    def anova_regression_table(self, result):
        return pd.DataFrame({
            "source": ["Regression", "Residual", "Total"],
            "SS": [result["ssr"], result["sse"], result["sst"]],
            "df": [result["df_model"], result["df_resid"], result["n"] - 1],
            "MS": [result["ssr"] / result["df_model"], result["mse"], np.nan],
            "F": [result["overall_f"], np.nan, np.nan],
            "p_value": [result["overall_f_p_value"], np.nan, np.nan],
        })

    def predict(self, result, new_data, confidence=0.95, interval="mean"):
        df = pd.DataFrame(new_data)
        Xnew = np.column_stack([np.ones(len(df)), df[result["x_cols"]].to_numpy(float)])
        pred = Xnew @ result["coefficients"]
        h = np.sum((Xnew @ np.linalg.inv(result["X"].T @ result["X"])) * Xnew, axis=1)
        mult = 1 if interval == "mean" else 1 + h
        if interval == "mean":
            se = np.sqrt(result["mse"] * h)
        elif interval == "prediction":
            se = np.sqrt(result["mse"] * mult)
        else:
            raise ValueError("interval must be 'mean' or 'prediction'.")
        q = stats.t(df=result["df_resid"]).ppf(1 - (1 - confidence) / 2)
        return pd.DataFrame({"prediction": pred, "se": se, "lower": pred - q * se, "upper": pred + q * se})

    def calculate_residuals(self, result):
        return pd.Series(result["residuals"], name="residual")

    def calculate_r_squared(self, result):
        return result["r_squared"]

    def overall_f_test(self, result, alpha=0.05, label=None):
        crit = stats.f(result["df_model"], result["df_resid"]).ppf(1 - alpha)
        return {"label": label or "overall regression F-test", "method": "overall F-test", "statistic": result["overall_f"],
                "critical_value": crit, "p_value": result["overall_f_p_value"], "alpha": alpha,
                "df": (result["df_model"], result["df_resid"]),
                "decision": "Reject H0" if result["overall_f_p_value"] < alpha else "Fail to reject H0"}

    def coefficient_t_test(self, result, coefficient, null_value=0, alternative="two-sided", alpha=0.05, label=None):
        idx = self._coef_index(result, coefficient)
        c = np.zeros(result["p"])
        c[idx] = 1
        return self.linear_combination_test(result, c, null_value, alternative, alpha, label or f"{coefficient} test")

    def linear_combination_test(self, result, contrast_vector, null_value=0, alternative="two-sided", alpha=0.05, label=None):
        alternative = self._check_alt(alternative)
        c = np.asarray(contrast_vector, dtype=float)
        if len(c) != result["p"]:
            raise ValueError(f"Contrast length must equal {result['p']} coefficients: {result['coefficient_names']}")
        estimate = float(c @ result["coefficients"])
        se = float(np.sqrt(c @ result["cov_beta"] @ c))
        statistic = (estimate - null_value) / se
        dist = stats.t(df=result["df_resid"])
        p_value = self._p_value(dist, statistic, alternative)
        crit = self._critical_values(dist, alpha, alternative)
        q = dist.ppf(1 - alpha / 2)
        ci = (estimate - q * se, estimate + q * se)
        return {"label": label or "linear combination t-test", "method": "linear combination t-test",
                "coefficient_names": result["coefficient_names"], "contrast_vector": c.tolist(),
                "estimate": estimate, "null_value": null_value, "standard_error": se, "statistic": float(statistic),
                "distribution": "t", "df": result["df_resid"], "alternative": alternative, "alpha": alpha,
                "p_value": p_value, "critical_values": crit, "confidence_interval": ci,
                "decision": "Reject H0" if p_value < alpha else "Fail to reject H0"}

    def linear_combination_ci(self, result, contrast_vector, confidence=0.95, label=None):
        test = self.linear_combination_test(result, contrast_vector, alpha=1 - confidence, label=label)
        return {"label": test["label"], "estimate": test["estimate"], "standard_error": test["standard_error"],
                "confidence_interval": test["confidence_interval"], "contrast_vector": test["contrast_vector"]}

    def general_linear_f_test(self, result, C, d=None, alpha=0.05, label=None):
        C = np.asarray(C, dtype=float)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        if C.shape[1] != result["p"]:
            raise ValueError(f"C must have {result['p']} columns.")
        q = C.shape[0]
        d = np.zeros(q) if d is None else np.asarray(d, dtype=float)
        diff = C @ result["coefficients"] - d
        middle = np.linalg.inv(C @ result["cov_beta"] @ C.T)
        f_stat = float((diff.T @ middle @ diff) / q)
        p_value = float(stats.f.sf(f_stat, q, result["df_resid"]))
        critical = float(stats.f.ppf(1 - alpha, q, result["df_resid"]))
        return {"label": label or "general linear F-test", "method": "general linear F-test", "C": C,
                "d": d, "statistic": f_stat, "critical_value": critical, "p_value": p_value,
                "df": (q, result["df_resid"]), "alpha": alpha,
                "decision": "Reject H0" if p_value < alpha else "Fail to reject H0"}

    def plot_simple_regression(self, result):
        if len(result["x_cols"]) != 1:
            raise ValueError("Simple regression plot requires exactly one predictor.")
        x_col, y_col = result["x_cols"][0], result["y_col"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(result["data"][x_col], result["y"], color="#1f4e79", alpha=0.8)
        order = np.argsort(result["data"][x_col].to_numpy())
        ax.plot(result["data"][x_col].to_numpy()[order], result["fitted"][order], color="#c00000", lw=2)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} fitted on {x_col}")
        fig.tight_layout()
        return fig, ax

    def plot_observed_vs_fitted(self, result):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(result["fitted"], result["y"], color="#1f4e79")
        lo, hi = min(result["fitted"].min(), result["y"].min()), max(result["fitted"].max(), result["y"].max())
        ax.plot([lo, hi], [lo, hi], color="black", ls="--")
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Observed")
        ax.set_title("Observed vs fitted")
        fig.tight_layout()
        return fig, ax

    def plot_residuals_vs_fitted(self, result):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(result["fitted"], result["residuals"], color="#1f4e79")
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs fitted")
        fig.tight_layout()
        return fig, ax

    def plot_qq_residuals(self, result):
        fig, ax = plt.subplots(figsize=(6, 5))
        stats.probplot(result["residuals"], dist="norm", plot=ax)
        ax.set_title("Residual normal Q-Q plot")
        fig.tight_layout()
        return fig, ax

    def plot_t_distribution_for_coefficient(self, test_result):
        return self.plot_t_distribution_for_linear_combination(test_result)

    def plot_t_distribution_for_linear_combination(self, test_result):
        dist = stats.t(df=test_result["df"])
        crit = test_result["critical_values"]
        vals = [test_result["statistic"]] + [v for v in crit if v is not None]
        xs = np.linspace(min(dist.ppf(0.001), min(vals) - 1), max(dist.ppf(0.999), max(vals) + 1), 700)
        ys = dist.pdf(xs)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ys, color="#1f4e79", label=f"t({test_result['df']}) under H0")
        if test_result["alternative"] == "greater":
            ax.fill_between(xs, 0, ys, where=xs >= crit[0], color="#f4b183", alpha=0.7, label="Critical region")
        elif test_result["alternative"] == "less":
            ax.fill_between(xs, 0, ys, where=xs <= crit[1], color="#f4b183", alpha=0.7, label="Critical region")
        else:
            ax.fill_between(xs, 0, ys, where=(xs <= crit[0]) | (xs >= crit[1]), color="#f4b183", alpha=0.7,
                            label="Critical region")
        ax.axvline(test_result["statistic"], color="#c00000", lw=2, label=f"Observed t = {test_result['statistic']:.3f}")
        ax.set_title(test_result["label"])
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_f_distribution_for_model(self, test_result):
        df1, df2 = test_result["df"]
        dist = stats.f(df1, df2)
        hi = max(dist.ppf(0.995), test_result["critical_value"] * 1.2, test_result["statistic"] * 1.2)
        xs = np.linspace(0, hi, 700)
        ys = dist.pdf(xs)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ys, color="#1f4e79", label=f"F({df1}, {df2}) under H0")
        ax.fill_between(xs, 0, ys, where=xs >= test_result["critical_value"], color="#f4b183", alpha=0.7,
                        label="Critical region")
        ax.axvline(test_result["statistic"], color="#c00000", lw=2, label=f"Observed F = {test_result['statistic']:.3f}")
        ax.set_title(test_result["label"])
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def _df(self):
        if self.data is None:
            raise ValueError("Load data first.")
        return self.data

    @staticmethod
    def _check_alpha(alpha):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        return alpha

    @staticmethod
    def _check_alt(alternative):
        if alternative not in {"two-sided", "greater", "less"}:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
        return alternative

    @staticmethod
    def _p_value(dist, statistic, alternative):
        if alternative == "greater":
            return float(dist.sf(statistic))
        if alternative == "less":
            return float(dist.cdf(statistic))
        return float(2 * min(dist.cdf(statistic), dist.sf(statistic)))

    @staticmethod
    def _critical_values(dist, alpha, alternative):
        if alternative == "greater":
            return (float(dist.ppf(1 - alpha)), None)
        if alternative == "less":
            return (None, float(dist.ppf(alpha)))
        return (float(dist.ppf(alpha / 2)), float(dist.ppf(1 - alpha / 2)))

    @staticmethod
    def _coef_index(result, coefficient):
        if isinstance(coefficient, int):
            return coefficient
        if coefficient not in result["coefficient_names"]:
            raise KeyError(f"Unknown coefficient {coefficient}. Choices: {result['coefficient_names']}")
        return result["coefficient_names"].index(coefficient)
