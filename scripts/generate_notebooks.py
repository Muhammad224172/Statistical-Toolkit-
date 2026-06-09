from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def write_notebook(name: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (Statistical Toolkit)", "language": "python", "name": "statistical-toolkit"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / name).write_text(json.dumps(notebook, indent=1), encoding="utf-8")


COMMON = r"""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def find_data_file(file_name):
    candidates = [ROOT / file_name, ROOT / "data" / file_name]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {file_name}. Put it in the project root or data/ folder and include the extension."
    )

def save_plot(fig, file_name):
    path = PLOTS_DIR / file_name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path
"""


classical_cells = [
    md("# Classical Inference: Body Temperature\n\nEach test is an independent editable block. Change the null value, alternative, alpha, or columns, then rerun that block."),
    code(COMMON + r"""
from classical_toolbox import ClassicalToolbox

use_default_dataset = True
custom_file_name = "my_classical_data.csv"

data_path = ROOT / "data" / "body_temperature.csv" if use_default_dataset else find_data_file(custom_file_name)
tool = ClassicalToolbox(data_path)
df = tool.data.copy()

required_default_columns = ["temperature", "gender"]
if use_default_dataset:
    missing = [c for c in required_default_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Default dataset is missing required columns: {missing}")

print("Dataset:", data_path)
print("Shape:", df.shape)
display(df.head())
display(df.isna().sum().to_frame("missing_values"))
display(tool.summarize_data())
print("Columns:", list(df.columns))
"""),
    code(r"""
# Editable block: one-sample mean z/t test
column = "temperature"
null_value = 98.6
alternative = "two-sided"
alpha = 0.05
sigma = None  # set a known sigma for a z test, or keep None for a t test

result = tool.one_sample_mean_test(column, null_value, alternative, alpha, sigma, label="One-sample mean test")
display(tool.format_result(result))
fig, ax = tool.plot_test_distribution(result)
save_plot(fig, "classical_one_sample_mean_test.png")
fig, ax = tool.plot_confidence_interval(result)
save_plot(fig, "classical_one_sample_mean_ci_from_test.png")
"""),
    code(r"""
# Editable block: two-sample mean test
group_col = "gender"
value_col = "temperature"
group_a = sorted(df[group_col].dropna().unique())[0]
group_b = sorted(df[group_col].dropna().unique())[1]
null_difference = 0
alternative = "two-sided"
alpha = 0.05
equal_var = False

result = tool.two_sample_mean_test(group_col, value_col, group_a, group_b, null_difference, alternative, alpha, equal_var, label="Two-sample mean test")
display(tool.format_result(result))
fig, ax = tool.plot_test_distribution(result)
save_plot(fig, "classical_two_sample_mean_test.png")
fig, ax = tool.plot_confidence_interval(result)
save_plot(fig, "classical_two_sample_mean_ci.png")
"""),
    code(r"""
# Editable block: paired mean test
# Default data are not paired, so the demo creates an after column. For your data, set before_col and after_col to real paired columns.
if use_default_dataset and "temperature_after" not in df.columns:
    paired_shift = np.linspace(-0.05, 0.35, len(df))
    df["temperature_after"] = df["temperature"] + paired_shift
    tool.load_data(df)

before_col = "temperature"
after_col = "temperature_after"
null_difference = 0
alternative = "greater"
alpha = 0.05

result = tool.paired_mean_test(before_col, after_col, null_difference, alternative, alpha, label="Paired mean test")
display(tool.format_result(result))
fig, ax = tool.plot_test_distribution(result)
save_plot(fig, "classical_paired_mean_test.png")
fig, ax = tool.plot_confidence_interval(result)
save_plot(fig, "classical_paired_mean_ci.png")
"""),
    code(r"""
# Editable block: one-sample variance chi-square test
column = "temperature"
null_variance = 0.5 ** 2
alternative = "two-sided"
alpha = 0.05

result = tool.one_sample_variance_test(column, null_variance, alternative, alpha, label="One-sample variance chi-square test")
display(tool.format_result(result))
fig, ax = tool.plot_test_distribution(result)
save_plot(fig, "classical_one_sample_variance_chi_square_test.png")
fig, ax = tool.plot_confidence_interval(result)
save_plot(fig, "classical_one_sample_variance_ci.png")
"""),
    code(r"""
# Editable block: two-sample variance F test
group_col = "gender"
value_col = "temperature"
group_a = sorted(df[group_col].dropna().unique())[0]
group_b = sorted(df[group_col].dropna().unique())[1]
null_ratio = 1
alternative = "two-sided"
alpha = 0.05

result = tool.two_sample_variance_test(group_col, value_col, group_a, group_b, null_ratio, alternative, alpha, label="Two-sample variance F test")
display(tool.format_result(result))
fig, ax = tool.plot_test_distribution(result)
save_plot(fig, "classical_two_sample_variance_f_test.png")
fig, ax = tool.plot_confidence_interval(result)
save_plot(fig, "classical_two_sample_variance_ci.png")
"""),
    code(r"""
# Editable block: one-sample proportion test
# Default demo tests whether the proportion of female observations differs from 0.5.
success_col = "gender"
success_value = "F"
null_probability = 0.5
alternative = "two-sided"
alpha = 0.05

result = tool.one_sample_proportion_test(success_col, success_value, null_probability, alternative, alpha, label="One-sample proportion test")
display(tool.format_result(result))
fig, ax = tool.plot_test_distribution(result)
save_plot(fig, "classical_one_sample_proportion_test.png")
fig, ax = tool.plot_confidence_interval(result)
save_plot(fig, "classical_one_sample_proportion_ci.png")
"""),
    code(r"""
# Editable block: mean confidence interval
column = "temperature"
confidence = 0.95
sigma = None

ci = tool.confidence_interval_mean(column, confidence, sigma)
display(pd.DataFrame({"column": [column], "confidence": [confidence], "mean_ci": [ci]}))
"""),
    code(r"""
# Editable block: variance confidence interval
column = "temperature"
confidence = 0.95

ci = tool.confidence_interval_variance(column, confidence)
display(pd.DataFrame({"column": [column], "confidence": [confidence], "variance_ci": [ci]}))
"""),
]


anova_cells = [
    md("# ANOVA: Plant Growth\n\nEach method is its own editable block. Contrasts and weights must follow the printed group order."),
    code(COMMON + r"""
from anova_toolbox import ANOVAToolbox

use_default_dataset = True
custom_file_name = "my_anova_data.csv"

data_path = ROOT / "data" / "plant_growth.csv" if use_default_dataset else find_data_file(custom_file_name)
tool = ANOVAToolbox(data_path)
df = tool.data.copy()

group_col = "group"
value_col = "weight"
if group_col not in df.columns or value_col not in df.columns:
    raise ValueError(f"Set group_col and value_col to columns in your data. Available columns: {list(df.columns)}")
if pd.to_numeric(df[value_col], errors="coerce").dropna().empty:
    raise ValueError(f"{value_col} must contain numeric values.")
if df[group_col].dropna().nunique() < 2:
    raise ValueError(f"{group_col} must contain at least two groups.")

print("Dataset:", data_path)
print("Shape:", df.shape)
display(df.head())
display(df.isna().sum().to_frame("missing_values"))
display(tool.summarize_data(group_col, value_col))
print("Group order:", list(tool.group_summary(group_col, value_col).index))
"""),
    code(r"""
# Editable block: one-way ANOVA
group_col = "group"
value_col = "weight"
alpha = 0.05

result = tool.one_way_anova(group_col, value_col, alpha, label="One-way ANOVA")
display(tool.anova_table(result))
print(result["decision"])
fig, ax = tool.plot_group_boxplot(group_col, value_col)
save_plot(fig, "anova_group_boxplot.png")
fig, ax = tool.plot_group_means_ci(group_col, value_col)
save_plot(fig, "anova_group_means_ci.png")
fig, ax = tool.plot_f_distribution(result)
save_plot(fig, "anova_one_way_f_critical_region.png")
fig, ax = tool.plot_residual_diagnostics(result)
save_plot(fig, "anova_residuals_vs_fitted.png")
"""),
    code(r"""
# Editable block: two-way ANOVA without replication
# The default PlantGrowth data have one treatment factor. This block creates a block factor by observation order.
row_factor = "block"
col_factor = "group"
value_col = "weight"
alpha = 0.05

two_way_df = df.copy()
if row_factor not in two_way_df.columns:
    two_way_df[row_factor] = two_way_df.groupby(col_factor).cumcount() + 1
two_way_tool = ANOVAToolbox(two_way_df)
result = two_way_tool.two_way_anova_no_replication(row_factor, col_factor, value_col, alpha, label="Two-way ANOVA without replication")
display(result["anova_table"])
"""),
    code(r"""
# Editable block: ANOVA contrast test
group_col = "group"
value_col = "weight"
group_order = list(tool.group_summary(group_col, value_col).index)
print("Group order:", group_order)
contrast_vector = [-1, 0, 1]
null_value = 0
alternative = "two-sided"
alpha = 0.05

result = tool.contrast_test(group_col, value_col, contrast_vector, null_value, alternative, alpha, label="ANOVA contrast test")
display(pd.DataFrame([result]))
fig, ax = tool.plot_t_distribution_for_contrast(result)
save_plot(fig, "anova_contrast_t_critical_region.png")
"""),
    code(r"""
# Editable block: linear combination confidence interval
group_col = "group"
value_col = "weight"
group_order = list(tool.group_summary(group_col, value_col).index)
print("Group order:", group_order)
weights = [0.5, 0.5, 0]
confidence = 0.95

result = tool.linear_combination_ci(group_col, value_col, weights, confidence, label="ANOVA linear combination CI")
display(pd.DataFrame([result]))
"""),
    code(r"""
# Editable block: Bonferroni pairwise comparisons
group_col = "group"
value_col = "weight"
alpha = 0.05

display(tool.bonferroni_pairwise(group_col, value_col, alpha))
"""),
]


regression_cells = [
    md("# Linear Regression: Airquality\n\nAll tests are independent editable blocks. For vectors, follow the printed coefficient order."),
    code(COMMON + r"""
from regression_toolbox import RegressionToolbox

use_default_dataset = True
custom_file_name = "my_regression_data.csv"

data_path = ROOT / "data" / "airquality.csv" if use_default_dataset else find_data_file(custom_file_name)
tool = RegressionToolbox(data_path)
df = tool.data.copy()

y_col = "Ozone"
simple_x_col = "Temp"
x_cols = ["Temp", "Wind", "Solar.R"]
needed = list(dict.fromkeys([y_col, simple_x_col] + x_cols))
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Set y_col/simple_x_col/x_cols to columns in your data. Missing: {missing}. Available: {list(df.columns)}")
clean_rows = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
if len(clean_rows) <= len(x_cols) + 1:
    raise ValueError("Regression needs more complete numeric rows than coefficients.")

print("Dataset:", data_path)
print("Shape:", df.shape)
display(df.head())
display(df.isna().sum().to_frame("missing_values"))
display(tool.summarize_data(needed))
"""),
    code(r"""
# Editable block: simple linear regression fit
simple_x_col = "Temp"
y_col = "Ozone"
alpha = 0.05

simple = tool.fit_simple_linear_regression(simple_x_col, y_col, alpha)
print("Coefficient order:", simple["coefficient_names"])
display(tool.coefficient_table(simple))
display(tool.anova_regression_table(simple))
fig, ax = tool.plot_simple_regression(simple)
save_plot(fig, "regression_simple_fit.png")
fig, ax = tool.plot_observed_vs_fitted(simple)
save_plot(fig, "regression_simple_observed_vs_fitted.png")
fig, ax = tool.plot_residuals_vs_fitted(simple)
save_plot(fig, "regression_simple_residuals_vs_fitted.png")
"""),
    code(r"""
# Editable block: test intercept a
coefficient = "Intercept"
null_value = 0
alternative = "two-sided"
alpha = 0.05

test = tool.coefficient_t_test(simple, coefficient, null_value, alternative, alpha, label="Test intercept a")
display(pd.DataFrame([test]))
fig, ax = tool.plot_t_distribution_for_coefficient(test)
save_plot(fig, "regression_intercept_a_t_test.png")
"""),
    code(r"""
# Editable block: test slope beta
coefficient = simple_x_col
null_value = 0
alternative = "greater"
alpha = 0.05

test = tool.coefficient_t_test(simple, coefficient, null_value, alternative, alpha, label="Test slope beta")
display(pd.DataFrame([test]))
fig, ax = tool.plot_t_distribution_for_coefficient(test)
save_plot(fig, "regression_slope_beta_t_test.png")
"""),
    code(r"""
# Editable block: multiple linear regression fit
x_cols = ["Temp", "Wind", "Solar.R"]
y_col = "Ozone"
alpha = 0.05

model = tool.fit_multiple_linear_regression(x_cols, y_col, alpha)
print("Coefficient order:", model["coefficient_names"])
display(tool.coefficient_table(model))
display(tool.anova_regression_table(model))
fig, ax = tool.plot_observed_vs_fitted(model)
save_plot(fig, "regression_multiple_observed_vs_fitted.png")
fig, ax = tool.plot_residuals_vs_fitted(model)
save_plot(fig, "regression_multiple_residuals_vs_fitted.png")
"""),
    code(r"""
# Editable block: multiple regression coefficient test
coefficient = "Wind"
null_value = 0
alternative = "less"
alpha = 0.05

test = tool.coefficient_t_test(model, coefficient, null_value, alternative, alpha, label="Multiple regression coefficient test")
display(pd.DataFrame([test]))
fig, ax = tool.plot_t_distribution_for_coefficient(test)
save_plot(fig, "regression_multiple_coefficient_t_test.png")
"""),
    code(r"""
# Editable block: test part of beta using c'beta
print("Coefficient order:", model["coefficient_names"])
contrast_vector = [0, 1, 0, 0]
null_value = 0
alternative = "two-sided"
alpha = 0.05

test = tool.linear_combination_test(model, contrast_vector, null_value, alternative, alpha, label="Test c'beta")
display(pd.DataFrame([test]))
fig, ax = tool.plot_t_distribution_for_linear_combination(test)
save_plot(fig, "regression_linear_combination_t_test.png")
"""),
    code(r"""
# Editable block: confidence interval for c'beta
contrast_vector = [0, 1, 0, 0]
confidence = 0.95

ci = tool.linear_combination_ci(model, contrast_vector, confidence, label="Linear combination CI")
display(pd.DataFrame([ci]))
"""),
    code(r"""
# Editable block: general linear F-test C beta = d
print("Coefficient order:", model["coefficient_names"])
C = [[0, 0, 1, 0], [0, 0, 0, 1]]
d = [0, 0]
alpha = 0.05

test = tool.general_linear_f_test(model, C, d, alpha, label="General linear F-test for part of beta")
display(pd.DataFrame([test]))
fig, ax = tool.plot_f_distribution_for_model(test)
save_plot(fig, "regression_general_linear_f_test.png")
"""),
    code(r"""
# Editable block: overall regression F-test
alpha = 0.05

test = tool.overall_f_test(model, alpha, label="Overall regression F-test")
display(pd.DataFrame([test]))
fig, ax = tool.plot_f_distribution_for_model(test)
save_plot(fig, "regression_overall_f_test.png")
"""),
    code(r"""
# Editable block: mean response and prediction intervals
new_data = pd.DataFrame({"Temp": [70, 80, 90], "Wind": [8, 10, 12], "Solar.R": [150, 200, 250]})
confidence = 0.95

display(tool.predict(model, new_data, confidence, interval="mean").assign(interval="mean"))
display(tool.predict(model, new_data, confidence, interval="prediction").assign(interval="prediction"))
"""),
    code(r"""
# Editable block: Scheffe simultaneous mean response band
alpha = 0.05

band = tool.scheffe_mean_response_band(simple, alpha=alpha)
display(band.head())
fig, ax = tool.plot_scheffe_band(simple, band, title="Scheffe simultaneous mean response band")
save_plot(fig, "regression_scheffe_mean_response_band.png")
"""),
    code(r"""
# Editable block: Scheffe-style prediction-band variant
# This is labeled as a prediction-band variant because Scheffe's standard band is for the mean response.
alpha = 0.05

band = tool.scheffe_prediction_band(simple, alpha=alpha)
display(band.head())
fig, ax = tool.plot_scheffe_band(simple, band, title="Scheffe-style prediction-band variant")
save_plot(fig, "regression_scheffe_prediction_band_variant.png")
"""),
]


bayesian_cells = [
    md("# Bayesian Inference: Titanic\n\nThis notebook shows conjugate prior, likelihood, posterior, sequential updating, prior sensitivity, Monte Carlo sampling, importance sampling, rejection sampling, and a Dirichlet-Multinomial example."),
    code(COMMON + r"""
from bayesian_toolbox import BayesianToolbox

use_default_dataset = True
custom_file_name = "my_bayesian_data.csv"

data_path = ROOT / "data" / "titanic.csv" if use_default_dataset else find_data_file(custom_file_name)
tool = BayesianToolbox(data_path)
df = tool.data.copy()

success_col = "survived"
continuous_col = "age"
category_col = "pclass"
needed = [success_col, continuous_col, category_col]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Set success_col/continuous_col/category_col to columns in your data. Missing: {missing}. Available: {list(df.columns)}")
if df[success_col].dropna().nunique() != 2:
    raise ValueError("Beta-Binomial example needs a binary success column.")
if pd.to_numeric(df[continuous_col], errors="coerce").dropna().empty:
    raise ValueError("Normal-Normal example needs a numeric continuous column.")

print("Dataset:", data_path)
print("Shape:", df.shape)
display(df.head())
display(df.isna().sum().to_frame("missing_values"))
display(tool.summarize_data())
"""),
    code(r"""
# Editable block: Beta-Binomial conjugate update
success_col = "survived"
success_value = 1
alpha_prior = 1
beta_prior = 1
hypothesized_p = 0.5
credibility = 0.95

result = tool.beta_binomial_from_column(success_col, success_value, alpha_prior, beta_prior, hypothesized_p, label="Titanic survival probability")
print("Likelihood:", result["likelihood"])
print("Prior:", result["prior"])
print("Posterior:", result["posterior"])
display(pd.DataFrame([result]))
display(tool.posterior_summary_beta(result, credibility))
print(f"P(p > {hypothesized_p} | data) =", tool.posterior_probability_beta(result, hypothesized_p, "greater"))
fig, ax = tool.plot_beta_prior_likelihood_posterior(result)
save_plot(fig, "bayesian_beta_prior_likelihood_posterior.png")
fig, ax = tool.plot_beta_credible_interval(result, credibility)
save_plot(fig, "bayesian_beta_credible_interval.png")
"""),
    code(r"""
# Editable block: sequential Beta-Binomial updating
values = df[success_col].dropna().eq(success_value).astype(int).to_numpy()
batches = []
for batch in np.array_split(values, 4):
    batches.append((int(batch.sum()), int(len(batch))))

seq = tool.beta_sequential_updates(batches, alpha_prior, beta_prior, hypothesized_p)
display(seq["updates"])
one_shot = result
print("Sequential final alpha,beta:", seq["updates"][["alpha_after", "beta_after"]].iloc[-1].to_dict())
print("One-shot alpha,beta:", {"alpha_posterior": one_shot["alpha_posterior"], "beta_posterior": one_shot["beta_posterior"]})
assert np.isclose(seq["updates"]["alpha_after"].iloc[-1], one_shot["alpha_posterior"])
assert np.isclose(seq["updates"]["beta_after"].iloc[-1], one_shot["beta_posterior"])
fig, ax = tool.plot_beta_sequential_updates(seq)
save_plot(fig, "bayesian_beta_sequential_updates.png")
"""),
    code(r"""
# Editable block: impact of weak and strong priors
successes = result["successes"]
trials = result["trials"]
priors = [("Weak prior Beta(1,1)", 1, 1), ("Strong prior Beta(20,20)", 20, 20)]

comparison = tool.compare_beta_priors(successes, trials, priors, hypothesized_p)
display(comparison["summary"])
fig, ax = tool.plot_beta_prior_comparison(comparison)
save_plot(fig, "bayesian_beta_weak_vs_strong_prior.png")
"""),
    code(r"""
# Editable block: Normal-Normal conjugate mean with known variance
continuous_col = "age"
prior_mean = 30
prior_variance = 100
sigma_sq = float(pd.to_numeric(df[continuous_col], errors="coerce").dropna().var(ddof=1))
hypothesized_mean = 30
credibility = 0.95

normal_result = tool.normal_mean_known_variance_update(continuous_col, prior_mean, prior_variance, sigma_sq, hypothesized_mean, label="Posterior mean age")
print("Likelihood:", normal_result["likelihood"])
print("Prior:", normal_result["prior"])
print("Posterior:", normal_result["posterior"])
display(pd.DataFrame([normal_result]).drop(columns=["data"]))
display(tool.posterior_summary_normal(normal_result, credibility))
print(f"P(mu > {hypothesized_mean} | data) =", tool.posterior_probability_normal(normal_result, hypothesized_mean, "greater"))
fig, ax = tool.plot_normal_prior_likelihood_posterior(normal_result)
save_plot(fig, "bayesian_normal_prior_likelihood_posterior.png")
fig, ax = tool.plot_normal_credible_interval(normal_result, credibility)
save_plot(fig, "bayesian_normal_credible_interval.png")
"""),
    code(r"""
# Editable block: direct Monte Carlo posterior sampling
n_samples = 10000
threshold = hypothesized_p

samples = tool.sample_beta_posterior(result, n_samples=n_samples, random_state=42)
summary = tool.monte_carlo_beta_summary(samples, threshold)
display(summary)
exact_ci = tool.posterior_summary_beta(result).set_index("quantity").loc[["credible_low", "credible_high"], "value"].to_numpy()
fig, ax = tool.plot_posterior_samples(samples, credible_interval=exact_ci)
save_plot(fig, "bayesian_monte_carlo_beta_samples.png")
"""),
    code(r"""
# Editable block: Monte Carlo convergence
exact_mean = result["alpha_posterior"] / (result["alpha_posterior"] + result["beta_posterior"])
convergence = tool.beta_monte_carlo_convergence(samples, true_value=exact_mean)
display(convergence.tail())
fig, ax = tool.plot_monte_carlo_convergence(convergence)
save_plot(fig, "bayesian_monte_carlo_convergence.png")
"""),
    code(r"""
# Editable block: importance sampling
proposal_alpha = 2
proposal_beta = 2
n_samples = 10000

importance = tool.beta_importance_sampling(successes, trials, alpha_prior, beta_prior, proposal_alpha, proposal_beta, n_samples, random_state=42)
display(pd.DataFrame([{k: v for k, v in importance.items() if k not in ["samples", "weights"]}]))
fig, ax = tool.plot_importance_sampling(importance)
save_plot(fig, "bayesian_importance_sampling_weights.png")
"""),
    code(r"""
# Editable block: rejection sampling
proposal_alpha = 1
proposal_beta = 1
n_samples = 2000

rejection = tool.beta_rejection_sampling(result, proposal_alpha, proposal_beta, n_samples=n_samples, random_state=42)
display(pd.DataFrame([{k: v for k, v in rejection.items() if k not in ["accepted_samples", "proposal_trace"]}]))
fig, ax = tool.plot_rejection_sampling(rejection)
save_plot(fig, "bayesian_rejection_sampling_trace.png")
fig, ax = tool.plot_posterior_samples(rejection["accepted_samples"])
save_plot(fig, "bayesian_rejection_sampling_accepted_samples.png")
"""),
    code(r"""
# Editable block: Dirichlet-Multinomial conjugate update
category_col = "pclass"
categories = sorted(df[category_col].dropna().unique())
alpha_prior = np.ones(len(categories))
credibility = 0.95

dirichlet_result = tool.dirichlet_multinomial_update(category_col, alpha_prior, categories, label="Titanic class probabilities")
print("Likelihood:", dirichlet_result["likelihood"])
print("Prior:", dirichlet_result["prior"])
print("Posterior:", dirichlet_result["posterior"])
display(pd.DataFrame({
    "category": dirichlet_result["categories"],
    "count": dirichlet_result["counts"],
    "alpha_prior": dirichlet_result["alpha_prior"],
    "alpha_posterior": dirichlet_result["alpha_posterior"],
}))
display(tool.posterior_summary_dirichlet(dirichlet_result, credibility))
assert np.allclose(dirichlet_result["alpha_posterior"], dirichlet_result["alpha_prior"] + dirichlet_result["counts"])
fig, ax = tool.plot_dirichlet_posterior_probabilities(dirichlet_result, credibility)
save_plot(fig, "bayesian_dirichlet_posterior_probabilities.png")
if len(categories) == 3:
    fig, ax = tool.plot_dirichlet_samples_simplex(dirichlet_result)
    save_plot(fig, "bayesian_dirichlet_simplex_samples.png")
"""),
    code(r"""
# Editable block: optional Gamma-Poisson conjugate count example
# Here counts are the number of passengers in each class in this small Titanic dataset.
counts = pd.Series(df[category_col].dropna()).value_counts().to_numpy()
alpha_prior = 1
beta_prior = 1
hypothesized_rate = counts.mean()

gamma_result = tool.gamma_poisson_update(counts, alpha_prior, beta_prior, hypothesized_rate, label="Gamma-Poisson class-count rate")
print("Likelihood:", gamma_result["likelihood"])
print("Prior:", gamma_result["prior"])
print("Posterior:", gamma_result["posterior"])
display(pd.DataFrame([gamma_result]))
display(tool.posterior_summary_gamma(gamma_result))
fig, ax = tool.plot_gamma_prior_likelihood_posterior(gamma_result)
save_plot(fig, "bayesian_gamma_poisson_posterior.png")
"""),
]


write_notebook("01_normal_inference_body_temperature.ipynb", classical_cells)
write_notebook("02_anova_plant_growth.ipynb", anova_cells)
write_notebook("03_regression_airquality.ipynb", regression_cells)
write_notebook("04_bayesian_titanic.ipynb", bayesian_cells)
print("Generated clean notebooks in", NOTEBOOKS)
