from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class ClassicalToolbox:
    """Classical inference tools for normal populations and proportions."""

    def __init__(self, data: str | Path | pd.DataFrame | None = None):
        self.data: pd.DataFrame | None = None
        if data is not None:
            self.load_data(data)

    def load_data(self, path_or_dataframe: str | Path | pd.DataFrame) -> pd.DataFrame:
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

    def summarize_data(self, columns: list[str] | None = None) -> pd.DataFrame:
        df = self._df()
        if columns is not None:
            df = df[columns]
        return df.describe(include="all").T

    def clean_numeric(self, column: str, dropna: bool = True) -> pd.Series:
        df = self._df()
        if column not in df.columns:
            raise KeyError(f"Column '{column}' was not found.")
        values = pd.to_numeric(df[column], errors="coerce")
        if dropna:
            values = values.dropna()
        if values.empty:
            raise ValueError(f"Column '{column}' has no numeric observations.")
        return values

    def one_sample_mean_test(self, column, null_value, alternative="two-sided", alpha=0.05, sigma=None, label=None):
        alternative = self._check_alternative(alternative)
        alpha = self._check_alpha(alpha)
        x = self.clean_numeric(column)
        n = len(x)
        mean = x.mean()
        sd = x.std(ddof=1)
        if sigma is None:
            se = sd / np.sqrt(n)
            statistic = (mean - null_value) / se
            dist = stats.t(df=n - 1)
            distribution = "t"
            df = n - 1
        else:
            se = sigma / np.sqrt(n)
            statistic = (mean - null_value) / se
            dist = stats.norm()
            distribution = "z"
            df = None
        p_value = self._p_value(dist, statistic, alternative)
        crit = self._critical_values(dist, alpha, alternative)
        ci = self.confidence_interval_mean(column, 1 - alpha, sigma)
        return self._result(label, "one-sample mean test", alternative, alpha, statistic, p_value, crit, distribution,
                            df, null_value, mean, se, ci, {"n": n, "mean": mean, "sd": sd})

    def two_sample_mean_test(self, group_col, value_col, group_a, group_b, null_difference=0, alternative="two-sided",
                             alpha=0.05, equal_var=False, label=None):
        alternative = self._check_alternative(alternative)
        alpha = self._check_alpha(alpha)
        df0 = self._df()
        a = pd.to_numeric(df0.loc[df0[group_col] == group_a, value_col], errors="coerce").dropna()
        b = pd.to_numeric(df0.loc[df0[group_col] == group_b, value_col], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            raise ValueError("Each group needs at least two numeric observations.")
        n1, n2 = len(a), len(b)
        m1, m2 = a.mean(), b.mean()
        v1, v2 = a.var(ddof=1), b.var(ddof=1)
        estimate = m1 - m2
        if equal_var:
            sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
            se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
            dfree = n1 + n2 - 2
        else:
            se = np.sqrt(v1 / n1 + v2 / n2)
            dfree = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        statistic = (estimate - null_difference) / se
        dist = stats.t(df=dfree)
        p_value = self._p_value(dist, statistic, alternative)
        crit = self._critical_values(dist, alpha, alternative)
        margin = dist.ppf(1 - alpha / 2) * se
        ci = (estimate - margin, estimate + margin)
        return self._result(label, "two-sample mean test", alternative, alpha, statistic, p_value, crit, "t",
                            dfree, null_difference, estimate, se, ci,
                            {"group_a": group_a, "group_b": group_b, "n_a": n1, "n_b": n2, "mean_a": m1, "mean_b": m2})

    def paired_mean_test(self, before_col, after_col, null_difference=0, alternative="two-sided", alpha=0.05, label=None):
        df0 = self._df()[[before_col, after_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df0) < 2:
            raise ValueError("Paired test needs at least two complete pairs.")
        temp = ClassicalToolbox(pd.DataFrame({"difference": df0[after_col] - df0[before_col]}))
        return temp.one_sample_mean_test("difference", null_difference, alternative, alpha, label=label or "paired mean test")

    def one_sample_variance_test(self, column, null_variance, alternative="two-sided", alpha=0.05, label=None):
        alternative = self._check_alternative(alternative)
        alpha = self._check_alpha(alpha)
        if null_variance <= 0:
            raise ValueError("Null variance must be positive.")
        x = self.clean_numeric(column)
        n = len(x)
        sample_var = x.var(ddof=1)
        statistic = (n - 1) * sample_var / null_variance
        dist = stats.chi2(df=n - 1)
        p_value = self._p_value(dist, statistic, alternative)
        crit = self._critical_values(dist, alpha, alternative)
        ci = self.confidence_interval_variance(column, 1 - alpha)
        return self._result(label, "one-sample variance test", alternative, alpha, statistic, p_value, crit, "chi-square",
                            n - 1, null_variance, sample_var, None, ci, {"n": n, "sample_variance": sample_var})

    def two_sample_variance_test(self, group_col, value_col, group_a, group_b, null_ratio=1, alternative="two-sided",
                                 alpha=0.05, label=None):
        alternative = self._check_alternative(alternative)
        alpha = self._check_alpha(alpha)
        if null_ratio <= 0:
            raise ValueError("Null variance ratio must be positive.")
        df0 = self._df()
        a = pd.to_numeric(df0.loc[df0[group_col] == group_a, value_col], errors="coerce").dropna()
        b = pd.to_numeric(df0.loc[df0[group_col] == group_b, value_col], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            raise ValueError("Each group needs at least two numeric observations.")
        v1, v2 = a.var(ddof=1), b.var(ddof=1)
        statistic = (v1 / v2) / null_ratio
        dist = stats.f(dfn=len(a) - 1, dfd=len(b) - 1)
        p_value = self._p_value(dist, statistic, alternative)
        crit = self._critical_values(dist, alpha, alternative)
        estimate = v1 / v2
        ci = (estimate / dist.ppf(1 - alpha / 2), estimate / dist.ppf(alpha / 2))
        return self._result(label, "two-sample variance test", alternative, alpha, statistic, p_value, crit, "F",
                            (len(a) - 1, len(b) - 1), null_ratio, estimate, None, ci,
                            {"group_a": group_a, "group_b": group_b, "variance_a": v1, "variance_b": v2})

    def one_sample_proportion_test(self, success_col, success_value=1, null_probability=0.5, alternative="two-sided",
                                   alpha=0.05, label=None):
        alternative = self._check_alternative(alternative)
        alpha = self._check_alpha(alpha)
        if not 0 < null_probability < 1:
            raise ValueError("Null probability must be between 0 and 1.")
        s = self._df()[success_col].dropna()
        n = len(s)
        successes = (s == success_value).sum()
        phat = successes / n
        se0 = np.sqrt(null_probability * (1 - null_probability) / n)
        statistic = (phat - null_probability) / se0
        dist = stats.norm()
        p_value = self._p_value(dist, statistic, alternative)
        crit = self._critical_values(dist, alpha, alternative)
        se = np.sqrt(phat * (1 - phat) / n)
        ci = (phat - stats.norm.ppf(1 - alpha / 2) * se, phat + stats.norm.ppf(1 - alpha / 2) * se)
        return self._result(label, "one-sample proportion test", alternative, alpha, statistic, p_value, crit, "z",
                            None, null_probability, phat, se0, ci, {"n": n, "successes": int(successes), "phat": phat})

    def confidence_interval_mean(self, column, confidence=0.95, sigma=None):
        x = self.clean_numeric(column)
        alpha = 1 - confidence
        n = len(x)
        mean = x.mean()
        if sigma is None:
            se = x.std(ddof=1) / np.sqrt(n)
            q = stats.t(df=n - 1).ppf(1 - alpha / 2)
        else:
            se = sigma / np.sqrt(n)
            q = stats.norm.ppf(1 - alpha / 2)
        return (mean - q * se, mean + q * se)

    def confidence_interval_variance(self, column, confidence=0.95):
        x = self.clean_numeric(column)
        alpha = 1 - confidence
        n = len(x)
        s2 = x.var(ddof=1)
        return ((n - 1) * s2 / stats.chi2(df=n - 1).ppf(1 - alpha / 2),
                (n - 1) * s2 / stats.chi2(df=n - 1).ppf(alpha / 2))

    def plot_test_distribution(self, result: dict[str, Any]):
        dist = self._dist_from_result(result)
        statistic = result["statistic"]
        crit = result["critical_values"]
        xs = self._plot_range(dist, statistic, crit)
        ys = dist.pdf(xs)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ys, color="#1f4e79", lw=2, label=f"Null {result['distribution']} distribution")
        self._shade_critical(ax, dist, xs, crit, result["alternative"])
        ax.axvline(statistic, color="#c00000", lw=2, label=f"Observed statistic = {statistic:.3f}")
        ax.set_title(result["label"])
        ax.set_xlabel("Test statistic")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_confidence_interval(self, result):
        low, high = result["confidence_interval"]
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.hlines(1, low, high, color="#1f4e79", lw=5)
        ax.plot(result["estimate"], 1, "o", color="#c00000", label="Estimate")
        ax.axvline(result["null_value"], color="black", ls="--", label="Null value")
        ax.set_yticks([])
        ax.set_title(f"{result['label']} confidence interval")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def format_result(self, result: dict[str, Any]) -> pd.DataFrame:
        keys = ["method", "null_value", "estimate", "standard_error", "statistic", "distribution", "df", "p_value",
                "alpha", "decision", "confidence_interval"]
        return pd.DataFrame({k: [result.get(k)] for k in keys})

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
    def _check_alternative(alternative):
        allowed = {"two-sided", "greater", "less"}
        if alternative not in allowed:
            raise ValueError(f"alternative must be one of {sorted(allowed)}.")
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

    def _result(self, label, method, alternative, alpha, statistic, p_value, crit, distribution, df, null_value,
                estimate, se, ci, details):
        return {
            "label": label or method,
            "method": method,
            "alternative": alternative,
            "alpha": alpha,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "critical_values": crit,
            "distribution": distribution,
            "df": df,
            "null_value": null_value,
            "estimate": float(estimate),
            "standard_error": None if se is None else float(se),
            "confidence_interval": tuple(float(v) for v in ci),
            "decision": "Reject H0" if p_value < alpha else "Fail to reject H0",
            "details": details,
        }

    @staticmethod
    def _shade_critical(ax, dist, xs, crit, alternative):
        ys = dist.pdf(xs)
        if alternative == "greater":
            ax.fill_between(xs, 0, ys, where=xs >= crit[0], color="#f4b183", alpha=0.7, label="Critical region")
        elif alternative == "less":
            ax.fill_between(xs, 0, ys, where=xs <= crit[1], color="#f4b183", alpha=0.7, label="Critical region")
        else:
            ax.fill_between(xs, 0, ys, where=(xs <= crit[0]) | (xs >= crit[1]), color="#f4b183", alpha=0.7,
                            label="Critical region")

    @staticmethod
    def _plot_range(dist, statistic, crit):
        vals = [statistic] + [v for v in crit if v is not None]
        lo = min(dist.ppf(0.001), min(vals) - 1)
        hi = max(dist.ppf(0.999), max(vals) + 1)
        return np.linspace(lo, hi, 700)

    @staticmethod
    def _dist_from_result(result):
        d = result["distribution"].lower()
        if d in {"t"}:
            return stats.t(df=result["df"])
        if d in {"z"}:
            return stats.norm()
        if d in {"chi-square"}:
            return stats.chi2(df=result["df"])
        if d == "f":
            df1, df2 = result["df"]
            return stats.f(dfn=df1, dfd=df2)
        raise ValueError(f"Unknown distribution {result['distribution']}.")


NormalInferenceToolbox = ClassicalToolbox
