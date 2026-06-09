from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class ANOVAToolbox:
    """ANOVA tables, pairwise comparisons, and linear contrasts."""

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

    def summarize_data(self, group_col=None, value_col=None):
        df = self._df()
        if group_col and value_col:
            return self.group_summary(group_col, value_col)
        return df.describe(include="all").T

    def group_summary(self, group_col, value_col):
        df = self._clean(group_col, value_col)
        return df.groupby(group_col)[value_col].agg(["count", "mean", "std", "var", "min", "max"])

    def one_way_anova(self, group_col, value_col, alpha=0.05, label=None):
        alpha = self._check_alpha(alpha)
        df = self._clean(group_col, value_col)
        groups = [g for g, _ in df.groupby(group_col, sort=True)]
        values = [df.loc[df[group_col] == g, value_col].to_numpy(float) for g in groups]
        n = sum(len(v) for v in values)
        k = len(groups)
        if k < 2 or n <= k:
            raise ValueError("ANOVA needs at least two groups and residual degrees of freedom.")
        grand = df[value_col].mean()
        ss_between = sum(len(v) * (v.mean() - grand) ** 2 for v in values)
        ss_within = sum(((v - v.mean()) ** 2).sum() for v in values)
        df_between = k - 1
        df_within = n - k
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        f_stat = ms_between / ms_within
        p_value = float(stats.f.sf(f_stat, df_between, df_within))
        critical = float(stats.f.ppf(1 - alpha, df_between, df_within))
        table = pd.DataFrame({
            "source": ["Between groups", "Within groups", "Total"],
            "SS": [ss_between, ss_within, ss_between + ss_within],
            "df": [df_between, df_within, n - 1],
            "MS": [ms_between, ms_within, np.nan],
            "F": [f_stat, np.nan, np.nan],
            "p_value": [p_value, np.nan, np.nan],
        })
        return {
            "label": label or "one-way ANOVA",
            "method": "one-way ANOVA",
            "groups": groups,
            "group_col": group_col,
            "value_col": value_col,
            "alpha": alpha,
            "statistic": float(f_stat),
            "critical_value": critical,
            "p_value": p_value,
            "df": (df_between, df_within),
            "mse": float(ms_within),
            "anova_table": table,
            "decision": "Reject H0" if p_value < alpha else "Fail to reject H0",
            "data": df,
        }

    def two_way_anova_no_replication(self, row_factor, col_factor, value_col, alpha=0.05, label=None):
        alpha = self._check_alpha(alpha)
        df = self._df()[[row_factor, col_factor, value_col]].copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna()
        table = df.pivot(index=row_factor, columns=col_factor, values=value_col)
        if table.isna().any().any():
            raise ValueError("Two-way ANOVA without replication requires one observation per cell.")
        r, c = table.shape
        grand = table.to_numpy().mean()
        row_means = table.mean(axis=1)
        col_means = table.mean(axis=0)
        ss_rows = c * ((row_means - grand) ** 2).sum()
        ss_cols = r * ((col_means - grand) ** 2).sum()
        ss_total = ((table - grand) ** 2).to_numpy().sum()
        ss_error = ss_total - ss_rows - ss_cols
        df_rows, df_cols, df_error = r - 1, c - 1, (r - 1) * (c - 1)
        ms_rows, ms_cols, ms_error = ss_rows / df_rows, ss_cols / df_cols, ss_error / df_error
        f_rows, f_cols = ms_rows / ms_error, ms_cols / ms_error
        out = pd.DataFrame({
            "source": [row_factor, col_factor, "Error", "Total"],
            "SS": [ss_rows, ss_cols, ss_error, ss_total],
            "df": [df_rows, df_cols, df_error, r * c - 1],
            "MS": [ms_rows, ms_cols, ms_error, np.nan],
            "F": [f_rows, f_cols, np.nan, np.nan],
            "p_value": [stats.f.sf(f_rows, df_rows, df_error), stats.f.sf(f_cols, df_cols, df_error), np.nan, np.nan],
        })
        return {"label": label or "two-way ANOVA without replication", "anova_table": out, "alpha": alpha}

    def contrast_test(self, group_col, value_col, contrast_vector, null_value=0, alternative="two-sided", alpha=0.05,
                      label=None):
        alternative = self._check_alt(alternative)
        anova = self.one_way_anova(group_col, value_col, alpha=alpha)
        summary = self.group_summary(group_col, value_col)
        groups = list(summary.index)
        c = np.asarray(contrast_vector, dtype=float)
        if len(c) != len(groups):
            raise ValueError(f"Contrast length {len(c)} must match group count {len(groups)}: {groups}")
        if not np.isclose(c.sum(), 0):
            raise ValueError("ANOVA treatment contrasts should sum to 0.")
        means = summary["mean"].to_numpy(float)
        ns = summary["count"].to_numpy(float)
        estimate = float(c @ means)
        se = float(np.sqrt(anova["mse"] * np.sum(c ** 2 / ns)))
        statistic = (estimate - null_value) / se
        dist = stats.t(df=anova["df"][1])
        p_value = self._p_value(dist, statistic, alternative)
        critical = self._critical_values(dist, alpha, alternative)
        q = dist.ppf(1 - alpha / 2)
        ci = (estimate - q * se, estimate + q * se)
        return {"label": label or "ANOVA contrast", "method": "ANOVA contrast", "groups": groups,
                "contrast_vector": c.tolist(), "estimate": estimate, "null_value": null_value,
                "standard_error": se, "statistic": float(statistic), "distribution": "t", "df": anova["df"][1],
                "alternative": alternative, "alpha": alpha, "p_value": float(p_value), "critical_values": critical,
                "confidence_interval": ci, "decision": "Reject H0" if p_value < alpha else "Fail to reject H0"}

    def linear_combination_ci(self, group_col, value_col, weights, confidence=0.95, label=None):
        summary = self.group_summary(group_col, value_col)
        anova = self.one_way_anova(group_col, value_col, alpha=1 - confidence)
        w = np.asarray(weights, dtype=float)
        if len(w) != len(summary):
            raise ValueError("Weights length must match number of groups.")
        estimate = float(w @ summary["mean"].to_numpy(float))
        se = float(np.sqrt(anova["mse"] * np.sum(w ** 2 / summary["count"].to_numpy(float))))
        q = stats.t(df=anova["df"][1]).ppf(1 - (1 - confidence) / 2)
        return {"label": label or "linear combination CI", "groups": list(summary.index), "weights": w.tolist(),
                "estimate": estimate, "standard_error": se, "confidence_interval": (estimate - q * se, estimate + q * se)}

    def bonferroni_pairwise(self, group_col, value_col, alpha=0.05):
        summary = self.group_summary(group_col, value_col)
        anova = self.one_way_anova(group_col, value_col, alpha)
        groups = list(summary.index)
        rows = []
        m = len(groups) * (len(groups) - 1) / 2
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                mean_diff = summary.loc[g1, "mean"] - summary.loc[g2, "mean"]
                se = np.sqrt(anova["mse"] * (1 / summary.loc[g1, "count"] + 1 / summary.loc[g2, "count"]))
                t_stat = mean_diff / se
                p = min(1, 2 * stats.t(df=anova["df"][1]).sf(abs(t_stat)) * m)
                rows.append({"group_a": g1, "group_b": g2, "difference": mean_diff, "t": t_stat,
                             "bonferroni_p": p, "decision": "Reject H0" if p < alpha else "Fail to reject H0"})
        return pd.DataFrame(rows)

    def tukey_style_pairwise(self, group_col, value_col, alpha=0.05):
        return self.bonferroni_pairwise(group_col, value_col, alpha)

    def anova_table(self, result):
        return result["anova_table"]

    def plot_group_boxplot(self, group_col, value_col):
        df = self._clean(group_col, value_col)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        df.boxplot(column=value_col, by=group_col, ax=ax)
        ax.set_title(f"{value_col} by {group_col}")
        fig.suptitle("")
        fig.tight_layout()
        return fig, ax

    def plot_group_means_ci(self, group_col, value_col, confidence=0.95):
        summary = self.group_summary(group_col, value_col)
        alpha = 1 - confidence
        q = stats.t(df=summary["count"] - 1).ppf(1 - alpha / 2)
        err = q * summary["std"] / np.sqrt(summary["count"])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(summary.index.astype(str), summary["mean"], yerr=err, fmt="o", capsize=5, color="#1f4e79")
        ax.set_title(f"Group means with {confidence:.0%} confidence intervals")
        ax.set_ylabel(value_col)
        fig.tight_layout()
        return fig, ax

    def plot_f_distribution(self, result):
        df1, df2 = result["df"]
        dist = stats.f(df1, df2)
        hi = max(dist.ppf(0.995), result["statistic"] * 1.2, result["critical_value"] * 1.2)
        xs = np.linspace(0, hi, 700)
        ys = dist.pdf(xs)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ys, color="#1f4e79", lw=2, label=f"F({df1}, {df2}) under H0")
        ax.fill_between(xs, 0, ys, where=xs >= result["critical_value"], color="#f4b183", alpha=0.7,
                        label="Critical region")
        ax.axvline(result["statistic"], color="#c00000", lw=2, label=f"Observed F = {result['statistic']:.3f}")
        ax.set_title(result["label"])
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_t_distribution_for_contrast(self, result):
        dist = stats.t(df=result["df"])
        crit = result["critical_values"]
        vals = [result["statistic"]] + [v for v in crit if v is not None]
        xs = np.linspace(min(dist.ppf(0.001), min(vals) - 1), max(dist.ppf(0.999), max(vals) + 1), 700)
        ys = dist.pdf(xs)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ys, color="#1f4e79", label=f"t({result['df']}) under H0")
        if result["alternative"] == "greater":
            ax.fill_between(xs, 0, ys, where=xs >= crit[0], color="#f4b183", alpha=0.7, label="Critical region")
        elif result["alternative"] == "less":
            ax.fill_between(xs, 0, ys, where=xs <= crit[1], color="#f4b183", alpha=0.7, label="Critical region")
        else:
            ax.fill_between(xs, 0, ys, where=(xs <= crit[0]) | (xs >= crit[1]), color="#f4b183", alpha=0.7,
                            label="Critical region")
        ax.axvline(result["statistic"], color="#c00000", lw=2, label=f"Observed t = {result['statistic']:.3f}")
        ax.set_title(result["label"])
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_residual_diagnostics(self, result):
        df = result["data"].copy()
        means = df.groupby(result["group_col"])[result["value_col"]].transform("mean")
        residuals = df[result["value_col"]] - means
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(means, residuals, color="#1f4e79")
        axes[0].axhline(0, color="black", lw=1)
        axes[0].set_title("Residuals vs fitted group means")
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title("Normal Q-Q plot")
        fig.tight_layout()
        return fig, axes

    def _df(self):
        if self.data is None:
            raise ValueError("Load data first.")
        return self.data

    def _clean(self, group_col, value_col):
        df = self._df()[[group_col, value_col]].copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        return df.dropna()

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
