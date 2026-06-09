# Statistical Toolkit

A notebook-only statistical toolbox for ENS505 course topics. It contains four class-based Python modules and four demonstration notebooks that load real local datasets, summarize them, then run editable statistical blocks with calculations and diagrams.

Q-Q plots are intentionally excluded because they were not covered in the course material used for this project.

## What Is Included

- Classical inference for means, variances, proportions, chi-square tests, and F tests
- ANOVA tables, treatment contrasts, linear combinations, and Bonferroni comparisons
- Simple and multiple linear regression, coefficient tests, tests for parts of beta, general linear F-tests, prediction intervals, and Scheffe bands
- Bayesian conjugate inference, sequential updating, weak vs strong priors, Monte Carlo sampling, importance sampling, rejection sampling, Gamma-Poisson, and Dirichlet-Multinomial
- Automatic large PNG plot saving in `plots/`
- Default datasets plus custom dataset support

## First-Time Setup

Open PowerShell in this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name statistical-toolkit --display-name "Python (Statistical Toolkit)"
jupyter notebook
```

Then open the notebooks from the `notebooks/` folder.

When a notebook opens, select the kernel named `Python (Statistical Toolkit)`.

If you see an error mentioning NumPy 1.x/2.x, PyArrow, NumExpr, Bottleneck, or SciPy, you are probably using Anaconda or another Python environment instead of the project environment. Switch the notebook kernel to `Python (Statistical Toolkit)` and rerun the cells.

## Project Files

Core modules:

- `classical_toolbox.py`
- `anova_toolbox.py`
- `regression_toolbox.py`
- `bayesian_toolbox.py`

Canonical notebooks:

- `notebooks/01_normal_inference_body_temperature.ipynb`
- `notebooks/02_anova_plant_growth.ipynb`
- `notebooks/03_regression_airquality.ipynb`
- `notebooks/04_bayesian_titanic.ipynb`

Default datasets:

- `data/body_temperature.csv`
- `data/plant_growth.csv`
- `data/airquality.csv`
- `data/titanic.csv`

Maintenance script:

- `scripts/generate_notebooks.py` regenerates clean notebooks with no saved outputs.

## How To Use Custom Datasets

Each notebook starts with a setup block:

```python
use_default_dataset = True
custom_file_name = "my_data.csv"
```

To use your own dataset:

1. Put your file in the project root folder or in `data/`.
2. Use `.csv`, `.xlsx`, `.xls`, or `.json`.
3. Include the file extension in `custom_file_name`.
4. Change `use_default_dataset = False`.
5. Edit the column names in the setup block and later test blocks.

The notebooks validate common problems: missing columns, nonnumeric columns, insufficient groups, invalid binary outcomes, contrast length mismatches, invalid prior parameters, and regression designs with too few complete rows.

## Notebook Requirements

`01_normal_inference_body_temperature.ipynb`:

- Needs a numeric measurement column for mean and variance tests.
- Needs a group column with at least two groups for two-sample tests.
- Needs paired before/after columns for a real paired test.
- Default columns are `temperature` and `gender`.

`02_anova_plant_growth.ipynb`:

- Needs a categorical treatment column.
- Needs a numeric response column.
- Needs at least two groups.
- Default columns are `group` and `weight`.

`03_regression_airquality.ipynb`:

- Needs one numeric response column.
- Needs one or more numeric predictor columns.
- Needs more complete numeric rows than regression coefficients.
- Default response is `Ozone`; default predictors are `Temp`, `Wind`, and `Solar.R`.

`04_bayesian_titanic.ipynb`:

- Beta-Binomial blocks need a binary success column.
- Normal-Normal blocks need a numeric continuous column.
- Dirichlet-Multinomial blocks need a categorical column.
- Default columns are `survived`, `age`, and `pclass`.

## Editable Blocks

Every method is in its own notebook block. Edit the inputs at the top of a block, then rerun that block.

Classical example:

```python
null_value = 98.6
alternative = "two-sided"
alpha = 0.05
```

ANOVA contrast example:

```python
contrast_vector = [-1, 0, 1]
```

Regression linear combination example:

```python
contrast_vector = [0, 1, 0, 0]
```

For contrasts and linear combinations, always follow the printed group or coefficient order.

## Plots

Every plotting block saves a large PNG to:

```text
plots/
```

The default size is about `10 x 6` inches or larger, saved at `dpi=200`. Plot files are generated artifacts and are ignored by Git through `.gitignore`.

To regenerate plots, run the notebooks from top to bottom. You can delete `plots/` at any time; it will be recreated automatically.

## Bayesian Model Statements

The Bayesian notebook prints the conjugate model statements directly.

Beta-Binomial:

```text
Likelihood: X | p ~ Binomial(n, p)
Prior: p ~ Beta(alpha, beta)
Posterior: p | X ~ Beta(alpha + x, beta + n - x)
```

Normal-Normal mean with known variance:

```text
Likelihood: X_i | mu ~ Normal(mu, sigma^2)
Prior: mu ~ Normal(mu0, tau0^2)
Posterior: mu | x ~ Normal(mu_n, tau_n^2)
```

Dirichlet-Multinomial:

```text
Likelihood: counts | p ~ Multinomial(n, p)
Prior: p ~ Dirichlet(alpha)
Posterior: p | counts ~ Dirichlet(alpha + counts)
```

## Validation Checklist

Before presentation, run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m compileall classical_toolbox.py anova_toolbox.py regression_toolbox.py bayesian_toolbox.py
jupyter notebook
```

Then run each notebook from a clean kernel using `Python (Statistical Toolkit)`.

Expected checks:

- All four modules compile.
- All notebooks run from top to bottom.
- Classical, ANOVA, and regression tests show critical-region diagrams.
- Bayesian blocks show prior, likelihood, posterior, credible intervals, and hypothesized values where applicable.
- Sequential Bayesian updating equals the one-shot posterior.
- Dirichlet posterior parameters equal prior parameters plus observed category counts.
- No Q-Q plot functions or notebook calls remain.
