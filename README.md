# Statistical Toolbox

Notebook-only statistical toolbox for ENS505 course topics:

- significance testing for normal distributions
- ANOVA and treatment contrasts
- simple and multiple linear regression
- Bayesian conjugate inference

Each notebook loads a local dataset, shows a data summary, and then runs independent editable test blocks. Change the variables at the top of a block, rerun it, and the numerical result plus diagram update.

## Setup

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name statistical-toolkit --display-name "Python (Statistical Toolkit)"
jupyter notebook
```

Open the notebooks in the `notebooks/` folder.

When a notebook opens, use the kernel named `Python (Statistical Toolkit)`. This avoids accidentally running the notebooks with Anaconda or another Python installation.

If you see an error mentioning NumPy 1.x/2.x, PyArrow, NumExpr, Bottleneck, or SciPy, you are almost certainly using the wrong kernel. Switch the notebook kernel to `Python (Statistical Toolkit)`, then run the cells again.

## Files

Core classes:

- `classical_toolbox.py` contains `ClassicalToolbox`
- `anova_toolbox.py` contains `ANOVAToolbox`
- `regression_toolbox.py` contains `RegressionToolbox`
- `bayesian_toolbox.py` contains `BayesianToolbox`

Demonstrations:

- `notebooks/01_normal_inference_body_temperature.ipynb`
- `notebooks/02_anova_plant_growth.ipynb`
- `notebooks/03_regression_airquality.ipynb`
- `notebooks/04_bayesian_titanic.ipynb`

Local data:

- `data/body_temperature.csv`: body temperature teaching dataset
- `data/plant_growth.csv`: R `PlantGrowth` dataset
- `data/airquality.csv`: R `airquality` dataset excerpt
- `data/titanic.csv`: Titanic passenger manifest excerpt

## How To Use A Test Block

Each notebook starts with a dataset setup block. The default setting uses the included dataset:

```python
use_default_dataset = True
```

To use your own data:

1. Put your file in the project folder or in the `data/` folder.
2. Use one of these formats: `.csv`, `.xlsx`, `.xls`, or `.json`.
3. Change the setup block:

```python
use_default_dataset = False
custom_file_name = "my_data.csv"
```

The file name must include the extension. If the file is not found, the notebook will tell you to put it in the project folder or `data/` folder.

After loading, each notebook validates whether the dataset is suitable for the planned use case. If your column names are different, edit the setup block and the later test blocks to use your column names.

## Custom Dataset Requirements

For `01_normal_inference_body_temperature.ipynb`:

- Needs at least one numeric measurement column for mean/variance tests.
- Needs a grouping column with at least two groups for two-sample tests.
- Default columns are `temperature` and `gender`.

For `02_anova_plant_growth.ipynb`:

- Needs one categorical group/treatment column.
- Needs one numeric response column.
- Needs at least two groups, with at least two numeric observations per group.
- Default columns are `group` and `weight`.

For `03_regression_airquality.ipynb`:

- Needs one numeric response column.
- Needs one or more numeric predictor columns.
- Needs more complete numeric rows than the number of regression coefficients.
- Default response is `Ozone`; default predictors are `Temp`, `Wind`, and `Solar.R`.

For `04_bayesian_titanic.ipynb`:

- Beta-Binomial blocks need a binary success column.
- Normal-Normal blocks need a numeric continuous column.
- Subgroup examples need a categorical subgroup column.
- Default columns are `survived`, `age`, and `sex`.

## Editable Test Inputs

Each test block starts with editable inputs, for example:

```python
null_value = 98.6
alternative = "two-sided"
alpha = 0.05
```

For ANOVA and regression contrasts, the notebook prints the group or coefficient order first. The vector must follow that order.

Example ANOVA contrast:

```python
contrast_vector = [-1, 0, 1]
```

Example regression linear combination:

```python
contrast_vector = [0, 1, 0, 0]
```

## Bayesian Blocks

The Bayesian notebook explicitly prints the conjugate model:

```text
Likelihood: X | p ~ Binomial(n, p)
Prior: p ~ Beta(alpha, beta)
Posterior: p | X ~ Beta(alpha + x, beta + n - x)
```

and, for a normal mean with known variance:

```text
Likelihood: X_i | mu ~ Normal(mu, sigma^2)
Prior: mu ~ Normal(mu0, tau0^2)
Posterior: mu | x ~ Normal(mu_n, tau_n^2)
```

Plots show prior, likelihood, posterior, credible interval, and the hypothesized parameter value.

## Notes For Presentation

The calculations use explicit formulas with NumPy and SciPy rather than `statsmodels`. This makes the implementation easier to explain in terms of the course notes:

- test statistic
- null distribution
- critical region
- p-value
- confidence or credible interval
- decision or posterior probability statement

Classical inference uses rejection regions. Bayesian inference uses posterior distributions and credible regions, so the diagrams show where the hypothesized value falls under the posterior rather than a classical critical region.
