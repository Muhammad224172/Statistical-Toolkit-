from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class BayesianToolbox:
    """Conjugate Bayesian inference, Monte Carlo, and sampling demonstrations."""

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

    def beta_binomial_from_column(self, success_col, success_value=1, alpha_prior=1, beta_prior=1,
                                  hypothesized_p=None, label=None):
        s = self._df()[success_col].dropna()
        successes = int((s == success_value).sum())
        return self.beta_binomial_update(successes, len(s), alpha_prior, beta_prior, hypothesized_p, label)

    def beta_binomial_likelihood_grid(self, successes, trials, grid_size=1000):
        p = np.linspace(0.001, 0.999, grid_size)
        like = stats.binom.pmf(successes, trials, p)
        return pd.DataFrame({"p": p, "likelihood": like, "scaled_likelihood": like / like.max()})

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

    def beta_sequential_updates(self, batches, alpha_prior=1, beta_prior=1, hypothesized_p=None):
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("Beta prior parameters must be positive.")
        rows = []
        a, b = float(alpha_prior), float(beta_prior)
        total_successes = 0
        total_trials = 0
        for i, (successes, trials) in enumerate(batches, start=1):
            if trials <= 0 or successes < 0 or successes > trials:
                raise ValueError("Each batch must satisfy trials > 0 and 0 <= successes <= trials.")
            a_before, b_before = a, b
            a += successes
            b += trials - successes
            total_successes += successes
            total_trials += trials
            rows.append({"batch": i, "successes": successes, "trials": trials,
                         "alpha_before": a_before, "beta_before": b_before,
                         "alpha_after": a, "beta_after": b, "posterior_mean": a / (a + b)})
        one_shot = self.beta_binomial_update(total_successes, total_trials, alpha_prior, beta_prior, hypothesized_p)
        return {"updates": pd.DataFrame(rows), "one_shot": one_shot, "hypothesized_p": hypothesized_p}

    def compare_beta_priors(self, successes, trials, priors, hypothesized_p=None):
        rows = []
        results = []
        for name, alpha_prior, beta_prior in priors:
            result = self.beta_binomial_update(successes, trials, alpha_prior, beta_prior, hypothesized_p, label=name)
            results.append(result)
            summary = self.posterior_summary_beta(result)
            rows.append({
                "prior": name,
                "alpha_prior": alpha_prior,
                "beta_prior": beta_prior,
                "alpha_posterior": result["alpha_posterior"],
                "beta_posterior": result["beta_posterior"],
                "posterior_mean": float(summary.loc[summary["quantity"] == "posterior_mean", "value"].iloc[0]),
                "credible_low": float(summary.loc[summary["quantity"] == "credible_low", "value"].iloc[0]),
                "credible_high": float(summary.loc[summary["quantity"] == "credible_high", "value"].iloc[0]),
            })
        return {"summary": pd.DataFrame(rows), "results": results}

    def monte_carlo_beta_summary(self, samples, threshold=None):
        samples = np.asarray(samples, dtype=float)
        out = {
            "sample_size": len(samples),
            "mc_mean": float(samples.mean()),
            "mc_median": float(np.median(samples)),
            "mc_credible_low": float(np.quantile(samples, 0.025)),
            "mc_credible_high": float(np.quantile(samples, 0.975)),
        }
        if threshold is not None:
            out[f"Pr(p > {threshold})"] = float(np.mean(samples > threshold))
        return pd.DataFrame([out])

    def beta_monte_carlo_convergence(self, samples, true_value=None):
        samples = np.asarray(samples, dtype=float)
        running_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
        df = pd.DataFrame({"sample_size": np.arange(1, len(samples) + 1), "running_mean": running_mean})
        if true_value is not None:
            df["target"] = true_value
        return df

    def beta_importance_sampling(self, successes, trials, alpha_prior=1, beta_prior=1,
                                 proposal_alpha=2, proposal_beta=2, n_samples=10000, random_state=42):
        if min(alpha_prior, beta_prior, proposal_alpha, proposal_beta) <= 0:
            raise ValueError("All beta parameters must be positive.")
        rng = np.random.default_rng(random_state)
        samples = rng.beta(proposal_alpha, proposal_beta, n_samples)
        log_target = stats.beta(alpha_prior, beta_prior).logpdf(samples) + stats.binom.logpmf(successes, trials, samples)
        log_proposal = stats.beta(proposal_alpha, proposal_beta).logpdf(samples)
        log_w = log_target - log_proposal
        log_w -= np.max(log_w)
        weights = np.exp(log_w)
        weights /= weights.sum()
        ess = 1 / np.sum(weights ** 2)
        return {"samples": samples, "weights": weights, "estimate_mean": float(np.sum(weights * samples)),
                "effective_sample_size": float(ess), "proposal": (proposal_alpha, proposal_beta)}

    def beta_rejection_sampling(self, result, proposal_alpha=1, proposal_beta=1, envelope_multiplier=None,
                                n_samples=5000, random_state=42, grid_size=2000):
        a, b = result["alpha_posterior"], result["beta_posterior"]
        if min(a, b, proposal_alpha, proposal_beta) <= 0:
            raise ValueError("All beta parameters must be positive.")
        grid = np.linspace(0.001, 0.999, grid_size)
        ratio = stats.beta(a, b).pdf(grid) / stats.beta(proposal_alpha, proposal_beta).pdf(grid)
        m = float(np.nanmax(ratio) * 1.05) if envelope_multiplier is None else float(envelope_multiplier)
        rng = np.random.default_rng(random_state)
        accepted = []
        proposed = []
        attempts = 0
        while len(accepted) < n_samples:
            x = rng.beta(proposal_alpha, proposal_beta)
            accept_prob = stats.beta(a, b).pdf(x) / (m * stats.beta(proposal_alpha, proposal_beta).pdf(x))
            u = rng.random()
            proposed.append((x, accept_prob, u <= accept_prob))
            attempts += 1
            if u <= accept_prob:
                accepted.append(x)
        return {"accepted_samples": np.array(accepted),
                "proposal_trace": pd.DataFrame(proposed, columns=["sample", "accept_probability", "accepted"]),
                "attempts": attempts, "acceptance_rate": n_samples / attempts,
                "envelope_multiplier": m, "proposal": (proposal_alpha, proposal_beta)}

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

    def dirichlet_multinomial_update(self, category_col, alpha_prior=None, categories=None, label=None):
        s = self._df()[category_col].dropna()
        if s.empty:
            raise ValueError("Category column has no observations.")
        if categories is None:
            categories = sorted(s.unique())
        counts = np.array([(s == cat).sum() for cat in categories], dtype=float)
        if alpha_prior is None:
            alpha_prior = np.ones(len(categories))
        alpha_prior = np.asarray(alpha_prior, dtype=float)
        if len(alpha_prior) != len(categories):
            raise ValueError("alpha_prior length must match number of categories.")
        if np.any(alpha_prior <= 0):
            raise ValueError("Dirichlet prior parameters must be positive.")
        alpha_posterior = alpha_prior + counts
        return {"label": label or "Dirichlet-Multinomial conjugate update",
                "model": "Dirichlet-Multinomial",
                "likelihood": "counts | p ~ Multinomial(n, p)",
                "prior": "p ~ Dirichlet(alpha)",
                "posterior": "p | counts ~ Dirichlet(alpha + counts)",
                "category_col": category_col, "categories": list(categories),
                "counts": counts, "alpha_prior": alpha_prior, "alpha_posterior": alpha_posterior}

    def posterior_summary_dirichlet(self, result, credibility=0.95, n_samples=20000, random_state=42):
        rng = np.random.default_rng(random_state)
        samples = rng.dirichlet(result["alpha_posterior"], size=n_samples)
        rows = []
        for i, category in enumerate(result["categories"]):
            rows.append({"category": category, "count": result["counts"][i],
                         "alpha_prior": result["alpha_prior"][i],
                         "alpha_posterior": result["alpha_posterior"][i],
                         "posterior_mean_probability": result["alpha_posterior"][i] / result["alpha_posterior"].sum(),
                         "credible_low": np.quantile(samples[:, i], (1 - credibility) / 2),
                         "credible_high": np.quantile(samples[:, i], 1 - (1 - credibility) / 2)})
        return pd.DataFrame(rows)

    def sample_dirichlet_posterior(self, result, n_samples=10000, random_state=42):
        rng = np.random.default_rng(random_state)
        return rng.dirichlet(result["alpha_posterior"], size=n_samples)

    def plot_beta_prior_likelihood_posterior(self, result):
        grid = np.linspace(0.001, 0.999, 1000)
        prior = stats.beta(result["alpha_prior"], result["beta_prior"]).pdf(grid)
        posterior = stats.beta(result["alpha_posterior"], result["beta_posterior"]).pdf(grid)
        like = stats.binom.pmf(result["successes"], result["trials"], grid)
        like_scaled = like / like.max() * max(prior.max(), posterior.max())
        fig, ax = plt.subplots(figsize=(10, 6))
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
        fig, ax = plt.subplots(figsize=(10, 6))
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

    def plot_beta_sequential_updates(self, sequential_result):
        updates = sequential_result["updates"]
        grid = np.linspace(0.001, 0.999, 1000)
        fig, ax = plt.subplots(figsize=(10, 6))
        for _, row in updates.iterrows():
            ax.plot(grid, stats.beta(row["alpha_after"], row["beta_after"]).pdf(grid),
                    lw=2, label=f"After batch {int(row['batch'])}")
        if sequential_result.get("hypothesized_p") is not None:
            ax.axvline(sequential_result["hypothesized_p"], color="black", ls="--", label="hypothesized p")
        ax.set_xlabel("p")
        ax.set_ylabel("Posterior density")
        ax.set_title("Sequential Beta-Binomial updating")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_beta_prior_comparison(self, comparison):
        grid = np.linspace(0.001, 0.999, 1000)
        fig, ax = plt.subplots(figsize=(10, 6))
        for result in comparison["results"]:
            ax.plot(grid, stats.beta(result["alpha_posterior"], result["beta_posterior"]).pdf(grid),
                    lw=2, label=result["label"])
        ax.set_xlabel("p")
        ax.set_ylabel("Posterior density")
        ax.set_title("Impact of weak and strong priors")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_normal_prior_likelihood_posterior(self, result):
        prior_sd = np.sqrt(result["prior_variance"])
        post_sd = np.sqrt(result["posterior_variance"])
        like_sd = np.sqrt(result["sigma_sq"] / result["n"])
        lo = min(result["prior_mean"] - 4 * prior_sd, result["sample_mean"] - 4 * like_sd,
                 result["posterior_mean"] - 4 * post_sd)
        hi = max(result["prior_mean"] + 4 * prior_sd, result["sample_mean"] + 4 * like_sd,
                 result["posterior_mean"] + 4 * post_sd)
        grid = np.linspace(lo, hi, 1000)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(grid, stats.norm(result["prior_mean"], prior_sd).pdf(grid), label=result["prior"], color="#1f4e79", lw=2)
        ax.plot(grid, stats.norm(result["sample_mean"], like_sd).pdf(grid), label="likelihood for sample mean",
                color="#70ad47", lw=2)
        ax.plot(grid, stats.norm(result["posterior_mean"], post_sd).pdf(grid), label=result["posterior"],
                color="#c00000", lw=2)
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
        fig, ax = plt.subplots(figsize=(10, 6))
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

    def plot_gamma_prior_likelihood_posterior(self, result):
        prior = stats.gamma(a=result["alpha_prior"], scale=1 / result["beta_prior"])
        post = stats.gamma(a=result["alpha_posterior"], scale=1 / result["beta_posterior"])
        hi = max(prior.ppf(0.995), post.ppf(0.995))
        grid = np.linspace(0.001, hi, 1000)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(grid, prior.pdf(grid), color="#1f4e79", label=result["prior"], lw=2)
        ax.plot(grid, post.pdf(grid), color="#c00000", label=result["posterior"], lw=2)
        if result.get("hypothesized_rate") is not None:
            ax.axvline(result["hypothesized_rate"], color="black", ls="--", label="hypothesized rate")
        ax.legend()
        ax.set_title(result["label"])
        fig.tight_layout()
        return fig, ax

    def plot_posterior_samples(self, samples, credible_interval=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(samples, bins=35, density=True, color="#9dc3e6", edgecolor="white")
        if credible_interval is not None:
            ax.axvline(credible_interval[0], color="black", ls="--")
            ax.axvline(credible_interval[1], color="black", ls="--")
        ax.set_title("Posterior samples")
        fig.tight_layout()
        return fig, ax

    def plot_monte_carlo_convergence(self, convergence_df):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(convergence_df["sample_size"], convergence_df["running_mean"], color="#1f4e79", lw=2)
        if "target" in convergence_df.columns:
            ax.axhline(convergence_df["target"].iloc[0], color="#c00000", ls="--", label="Exact posterior mean")
            ax.legend()
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("Running estimate")
        ax.set_title("Monte Carlo convergence")
        fig.tight_layout()
        return fig, ax

    def plot_importance_sampling(self, importance_result):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(importance_result["samples"], importance_result["weights"], s=8, alpha=0.35, color="#1f4e79")
        ax.set_xlabel("Proposal sample")
        ax.set_ylabel("Normalized importance weight")
        ax.set_title(f"Importance sampling weights; ESS = {importance_result['effective_sample_size']:.1f}")
        fig.tight_layout()
        return fig, ax

    def plot_rejection_sampling(self, rejection_result):
        trace = rejection_result["proposal_trace"]
        fig, ax = plt.subplots(figsize=(10, 6))
        accepted = trace[trace["accepted"]]
        rejected = trace[~trace["accepted"]]
        ax.scatter(rejected["sample"], rejected["accept_probability"], s=8, alpha=0.25, label="Rejected", color="#999999")
        ax.scatter(accepted["sample"], accepted["accept_probability"], s=8, alpha=0.45, label="Accepted", color="#1f4e79")
        ax.set_xlabel("Proposal sample")
        ax.set_ylabel("Acceptance probability")
        ax.set_title(f"Rejection sampling; acceptance rate = {rejection_result['acceptance_rate']:.3f}")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_dirichlet_posterior_probabilities(self, result, credibility=0.95):
        summary = self.posterior_summary_dirichlet(result, credibility)
        yerr = np.vstack([summary["posterior_mean_probability"] - summary["credible_low"],
                          summary["credible_high"] - summary["posterior_mean_probability"]])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(summary["category"].astype(str), summary["posterior_mean_probability"], color="#1f4e79", alpha=0.75)
        ax.errorbar(summary["category"].astype(str), summary["posterior_mean_probability"], yerr=yerr,
                    fmt="none", ecolor="black", capsize=5)
        ax.set_ylabel("Posterior category probability")
        ax.set_title(result["label"])
        fig.tight_layout()
        return fig, ax

    def plot_dirichlet_samples_simplex(self, result, n_samples=3000, random_state=42):
        if len(result["categories"]) != 3:
            raise ValueError("Simplex plot requires exactly three categories.")
        samples = self.sample_dirichlet_posterior(result, n_samples, random_state)
        x = samples[:, 1] + 0.5 * samples[:, 2]
        y = (np.sqrt(3) / 2) * samples[:, 2]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(x, y, s=8, alpha=0.25, color="#1f4e79")
        triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
        ax.plot(triangle[:, 0], triangle[:, 1], color="black")
        ax.text(-0.04, -0.04, str(result["categories"][0]))
        ax.text(1.02, -0.04, str(result["categories"][1]))
        ax.text(0.48, np.sqrt(3) / 2 + 0.03, str(result["categories"][2]))
        ax.set_title("Dirichlet posterior samples on 3-category simplex")
        ax.axis("off")
        fig.tight_layout()
        return fig, ax

    def _df(self):
        if self.data is None:
            raise ValueError("Load data first.")
        return self.data
