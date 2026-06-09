from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class BayesianToolbox:
    """Conjugate Bayesian inference with prior, likelihood, posterior, and plots."""

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

    def beta_binomial_update(self, successes, trials, alpha_prior=1, beta_prior=1, hypothesized_p=None, label=None):
        if trials <= 0 or successes < 0 or successes > trials:
            raise ValueError("Use trials > 0 and 0 <= successes <= trials.")
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("Beta prior parameters must be positive.")
        a_post = alpha_prior + successes
        b_post = beta_prior + trials - successes
        return {
            "label": label or "Beta-Binomial conjugate update",
            "model": "Beta-Binomial",
            "likelihood": "X | p ~ Binomial(n, p)",
            "prior": "p ~ Beta(alpha, beta)",
            "posterior": "p | X ~ Beta(alpha + x, beta + n - x)",
            "successes": int(successes),
            "trials": int(trials),
            "alpha_prior": float(alpha_prior),
            "beta_prior": float(beta_prior),
            "alpha_posterior": float(a_post),
            "beta_posterior": float(b_post),
            "hypothesized_p": hypothesized_p,
        }

    def beta_binomial_from_column(self, success_col, success_value=1, alpha_prior=1, beta_prior=1, hypothesized_p=None,
                                  label=None):
        s = self._df()[success_col].dropna()
        successes = int((s == success_value).sum())
        return self.beta_binomial_update(successes, len(s), alpha_prior, beta_prior, hypothesized_p, label)

    def beta_binomial_likelihood_grid(self, successes, trials, grid_size=1000):
        p = np.linspace(0.001, 0.999, grid_size)
        like = stats.binom.pmf(successes, trials, p)
        like_scaled = like / like.max()
        return pd.DataFrame({"p": p, "likelihood": like, "scaled_likelihood": like_scaled})

    def posterior_summary_beta(self, result, credibility=0.95):
        a, b = result["alpha_posterior"], result["beta_posterior"]
        low, high = stats.beta(a, b).ppf([(1 - credibility) / 2, 1 - (1 - credibility) / 2])
        mode = np.nan if a <= 1 or b <= 1 else (a - 1) / (a + b - 2)
        return pd.DataFrame({
            "quantity": ["posterior_mean", "posterior_median", "posterior_mode", "credible_low", "credible_high"],
            "value": [a / (a + b), stats.beta(a, b).median(), mode, low, high],
        })

    def posterior_probability_beta(self, result, threshold, direction="greater"):
        dist = stats.beta(result["alpha_posterior"], result["beta_posterior"])
        if direction == "greater":
            return float(dist.sf(threshold))
        if direction == "less":
            return float(dist.cdf(threshold))
        raise ValueError("direction must be 'greater' or 'less'.")

    def sample_beta_posterior(self, result, n_samples=10000, random_state=42):
        rng = np.random.default_rng(random_state)
        return rng.beta(result["alpha_posterior"], result["beta_posterior"], n_samples)

    def plot_beta_prior_likelihood_posterior(self, result):
        grid = np.linspace(0.001, 0.999, 1000)
        prior = stats.beta(result["alpha_prior"], result["beta_prior"]).pdf(grid)
        posterior = stats.beta(result["alpha_posterior"], result["beta_posterior"]).pdf(grid)
        like = stats.binom.pmf(result["successes"], result["trials"], grid)
        like_scaled = like / like.max() * max(prior.max(), posterior.max())
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(grid, prior, label=result["prior"], color="#1f4e79", lw=2)
        ax.plot(grid, like_scaled, label="scaled likelihood", color="#70ad47", lw=2)
        ax.plot(grid, posterior, label=result["posterior"], color="#c00000", lw=2)
        if result.get("hypothesized_p") is not None:
            ax.axvline(result["hypothesized_p"], color="black", ls="--",
                       label=f"hypothesized p = {result['hypothesized_p']}")
        ax.set_xlabel("p")
        ax.set_ylabel("Density / scaled likelihood")
        ax.set_title(result["label"])
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_beta_credible_interval(self, result, credibility=0.95):
        a, b = result["alpha_posterior"], result["beta_posterior"]
        dist = stats.beta(a, b)
        low, high = dist.ppf([(1 - credibility) / 2, 1 - (1 - credibility) / 2])
        grid = np.linspace(0.001, 0.999, 1000)
        pdf = dist.pdf(grid)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(grid, pdf, color="#c00000", lw=2, label="posterior")
        ax.fill_between(grid, 0, pdf, where=(grid >= low) & (grid <= high), color="#9dc3e6", alpha=0.7,
                        label=f"{credibility:.0%} credible interval")
        if result.get("hypothesized_p") is not None:
            ax.axvline(result["hypothesized_p"], color="black", ls="--", label="hypothesized value")
        ax.legend()
        ax.set_title("Beta posterior credible interval")
        fig.tight_layout()
        return fig, ax

    def plot_beta_hypothesized_value(self, result):
        return self.plot_beta_credible_interval(result)

    def normal_mean_known_variance_update(self, column, prior_mean, prior_variance, sigma_sq, hypothesized_mean=None,
                                          label=None):
        if prior_variance <= 0 or sigma_sq <= 0:
            raise ValueError("Prior variance and sigma_sq must be positive.")
        x = pd.to_numeric(self._df()[column], errors="coerce").dropna().to_numpy(float)
        n = len(x)
        if n == 0:
            raise ValueError("No numeric data available.")
        xbar = x.mean()
        posterior_variance = 1 / (1 / prior_variance + n / sigma_sq)
        posterior_mean = posterior_variance * (prior_mean / prior_variance + n * xbar / sigma_sq)
        return {
            "label": label or "Normal-Normal conjugate update",
            "model": "Normal-Normal",
            "likelihood": "X_i | mu ~ Normal(mu, sigma^2)",
            "prior": "mu ~ Normal(mu0, tau0^2)",
            "posterior": "mu | x ~ Normal(mu_n, tau_n^2)",
            "column": column,
            "n": n,
            "sample_mean": float(xbar),
            "sigma_sq": float(sigma_sq),
            "prior_mean": float(prior_mean),
            "prior_variance": float(prior_variance),
            "posterior_mean": float(posterior_mean),
            "posterior_variance": float(posterior_variance),
            "hypothesized_mean": hypothesized_mean,
            "data": x,
        }

    def normal_mean_likelihood_grid(self, column, sigma_sq, grid_size=1000):
        x = pd.to_numeric(self._df()[column], errors="coerce").dropna().to_numpy(float)
        se = np.sqrt(sigma_sq / len(x))
        center = x.mean()
        grid = np.linspace(center - 5 * se, center + 5 * se, grid_size)
        likelihood = stats.norm(center, se).pdf(grid)
        return pd.DataFrame({"mu": grid, "scaled_likelihood": likelihood / likelihood.max()})

    def posterior_summary_normal(self, result, credibility=0.95):
        sd = np.sqrt(result["posterior_variance"])
        low, high = stats.norm(result["posterior_mean"], sd).ppf([(1 - credibility) / 2, 1 - (1 - credibility) / 2])
        return pd.DataFrame({
            "quantity": ["posterior_mean", "posterior_sd", "credible_low", "credible_high"],
            "value": [result["posterior_mean"], sd, low, high],
        })

    def posterior_probability_normal(self, result, threshold, direction="greater"):
        dist = stats.norm(result["posterior_mean"], np.sqrt(result["posterior_variance"]))
        if direction == "greater":
            return float(dist.sf(threshold))
        if direction == "less":
            return float(dist.cdf(threshold))
        raise ValueError("direction must be 'greater' or 'less'.")

    def sample_normal_posterior(self, result, n_samples=10000, random_state=42):
        rng = np.random.default_rng(random_state)
        return rng.normal(result["posterior_mean"], np.sqrt(result["posterior_variance"]), n_samples)

    def plot_normal_prior_likelihood_posterior(self, result):
        prior_sd = np.sqrt(result["prior_variance"])
        post_sd = np.sqrt(result["posterior_variance"])
        like_sd = np.sqrt(result["sigma_sq"] / result["n"])
        lo = min(result["prior_mean"] - 4 * prior_sd, result["sample_mean"] - 4 * like_sd,
                 result["posterior_mean"] - 4 * post_sd)
        hi = max(result["prior_mean"] + 4 * prior_sd, result["sample_mean"] + 4 * like_sd,
                 result["posterior_mean"] + 4 * post_sd)
        grid = np.linspace(lo, hi, 1000)
        prior = stats.norm(result["prior_mean"], prior_sd).pdf(grid)
        like = stats.norm(result["sample_mean"], like_sd).pdf(grid)
        posterior = stats.norm(result["posterior_mean"], post_sd).pdf(grid)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(grid, prior, label=result["prior"], color="#1f4e79", lw=2)
        ax.plot(grid, like, label="likelihood for sample mean", color="#70ad47", lw=2)
        ax.plot(grid, posterior, label=result["posterior"], color="#c00000", lw=2)
        if result.get("hypothesized_mean") is not None:
            ax.axvline(result["hypothesized_mean"], color="black", ls="--",
                       label=f"hypothesized mean = {result['hypothesized_mean']}")
        ax.set_xlabel("mu")
        ax.set_ylabel("Density")
        ax.set_title(result["label"])
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_normal_credible_interval(self, result, credibility=0.95):
        sd = np.sqrt(result["posterior_variance"])
        dist = stats.norm(result["posterior_mean"], sd)
        low, high = dist.ppf([(1 - credibility) / 2, 1 - (1 - credibility) / 2])
        grid = np.linspace(result["posterior_mean"] - 4 * sd, result["posterior_mean"] + 4 * sd, 1000)
        pdf = dist.pdf(grid)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(grid, pdf, color="#c00000", lw=2, label="posterior")
        ax.fill_between(grid, 0, pdf, where=(grid >= low) & (grid <= high), color="#9dc3e6", alpha=0.7,
                        label=f"{credibility:.0%} credible interval")
        if result.get("hypothesized_mean") is not None:
            ax.axvline(result["hypothesized_mean"], color="black", ls="--", label="hypothesized mean")
        ax.set_title("Normal posterior credible interval")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_normal_hypothesized_mean(self, result):
        return self.plot_normal_credible_interval(result)

    def gamma_poisson_update(self, counts, alpha_prior=1, beta_prior=1, hypothesized_rate=None, label=None):
        counts = np.asarray(counts, dtype=float)
        if np.any(counts < 0) or alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("Counts must be nonnegative and Gamma prior parameters positive.")
        return {"label": label or "Gamma-Poisson conjugate update", "model": "Gamma-Poisson",
                "likelihood": "X_i | lambda ~ Poisson(lambda)", "prior": "lambda ~ Gamma(alpha, beta)",
                "posterior": "lambda | x ~ Gamma(alpha + sum(x), beta + n)",
                "alpha_prior": alpha_prior, "beta_prior": beta_prior,
                "alpha_posterior": alpha_prior + counts.sum(), "beta_posterior": beta_prior + len(counts),
                "hypothesized_rate": hypothesized_rate}

    def posterior_summary_gamma(self, result, credibility=0.95):
        dist = stats.gamma(a=result["alpha_posterior"], scale=1 / result["beta_posterior"])
        low, high = dist.ppf([(1 - credibility) / 2, 1 - (1 - credibility) / 2])
        return pd.DataFrame({"quantity": ["posterior_mean", "posterior_sd", "credible_low", "credible_high"],
                             "value": [dist.mean(), dist.std(), low, high]})

    def plot_gamma_prior_likelihood_posterior(self, result):
        prior = stats.gamma(a=result["alpha_prior"], scale=1 / result["beta_prior"])
        post = stats.gamma(a=result["alpha_posterior"], scale=1 / result["beta_posterior"])
        hi = max(prior.ppf(0.995), post.ppf(0.995))
        grid = np.linspace(0.001, hi, 1000)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(grid, prior.pdf(grid), color="#1f4e79", label=result["prior"], lw=2)
        ax.plot(grid, post.pdf(grid), color="#c00000", label=result["posterior"], lw=2)
        if result.get("hypothesized_rate") is not None:
            ax.axvline(result["hypothesized_rate"], color="black", ls="--", label="hypothesized rate")
        ax.legend()
        ax.set_title(result["label"])
        fig.tight_layout()
        return fig, ax

    def plot_posterior_samples(self, samples, credible_interval=None):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(samples, bins=35, density=True, color="#9dc3e6", edgecolor="white")
        if credible_interval is not None:
            ax.axvline(credible_interval[0], color="black", ls="--")
            ax.axvline(credible_interval[1], color="black", ls="--")
        ax.set_title("Posterior samples")
        fig.tight_layout()
        return fig, ax

    def _df(self):
        if self.data is None:
            raise ValueError("Load data first.")
        return self.data
