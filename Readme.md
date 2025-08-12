```markdown
# MLP Regression Project

A modular Python package for training, evaluating, and visualizing a multi‐layer perceptron (MLP) regression model with k‐fold cross‐validation and detailed diagnostics.

## Repository Structure



main/
├── data/                   # Raw and processed datasets
│   ├── cleaned-first/
│   ├── original-first/
│   ├── original-second/
│   └── splited-first/
├── notebook/               # Interactive Jupyter workflows
│   ├── 58c0fccc/
│   ├── Train.ipynb
│   ├── Test.ipynb
│   └── plot-sweep-samples.ipynb
└── src/
└── my\_package/         # Core package modules
├── **init**.py
├── config.py
├── preprocessing.py
├── model.py
├── training.py
├── io.py
├── plotting.py
├── evaluate.py
└── sweep.py
```
````

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/kamiaras/Acid-neural-net
   cd your-repo
````

2. **Create & activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   Create a `requirements.txt` listing:

   ```
   pandas
   numpy
   torch
   scikit-learn
   matplotlib
   ```

   Then install:

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install in editable mode**
   To import `my_package` directly:

   ```bash
   pip install -e src
   ```

## Quick Start

In a script or notebook, add the package root to `sys.path` (if not installed):

```python
import sys, os
sys.path.insert(0, os.path.abspath("src"))
```

Then import:

```python
from my_package import (
    cross_validate_mlp_with_history,
    plot_diagnostics,
    evaluate_and_plot_mlp,
    sweep_plot_samples_mlp
)
```

---

## Package Initialization (`src/my_package/__init__.py`)

Defines the public API for the package:

```python
"""
my_package: Public API for MLP workflows—configuration, model definition,
preprocessing, training, evaluation, and visualization.
"""

from .config       import categories_map, all_inputs, numeric_inputs
from .model        import MLP
from .preprocessing import prepare_data
from .training     import cross_validate_mlp_with_history
from .io           import make_tag, create_output_dir, save_models, save_hyperparams, save_norms
from .plotting     import plot_diagnostics
from .evaluate     import evaluate_and_plot_mlp
from .sweep        import sweep_plot_samples_mlp

__all__ = [
    'categories_map', 'all_inputs', 'numeric_inputs',
    'MLP', 'prepare_data', 'cross_validate_mlp_with_history',
    'make_tag', 'create_output_dir', 'save_models', 'save_hyperparams', 'save_norms',
    'plot_diagnostics', 'evaluate_and_plot_mlp', 'sweep_plot_samples_mlp'
]
```

---

## Configuration (`src/my_package/config.py`)

Centralizes feature definitions:

* **`categories_map`**: Allowed categories for Inputs 1–8.
* **`all_inputs`**: Inputs 1–18 (excluding 12).
* **`numeric_inputs`**: Inputs 9–18 (excluding 12).

---

## Preprocessing (`src/my_package/preprocessing.py`)

Full pipeline to clean, encode, and normalize data:

* `drop_unused(df, to_drop=('F2','F3'))`
* `normalize_target(y)`
* `encode_categoricals(X)`
* `normalize_numeric(X)`
* `prepare_data(df)` → `(X_np, y, norms, feature_names)`

Usage:

```python
from my_package.preprocessing import prepare_data
X_np, y, norms, feature_names = prepare_data(df)
```

---

## Model Definition (`src/my_package/model.py`)

Defines the `MLP` class:

```python
from torch import nn

class MLP(nn.Module):
    """
    in_dim: number of features
    layer_dims: list of hidden layer sizes
    activations: list of activation names ('relu','tanh','sigmoid','softplus')
    """
    ...
```

Example:

```python
from my_package.model import MLP
model = MLP(in_dim=16, layer_dims=[32,16], activations=['relu','tanh'])
```

---

## Training (`src/my_package/training.py`)

`cross_validate_mlp_with_history(df, k_folds, layer_dims, activations, epochs, lr, batch_size, device, random_state, loss_type)`

* Performs k‐fold training
* Returns:

  1. `fold_results`
  2. `histories`
  3. `models`
  4. `norms`
  5. `feature_names`

Example:

```python
from my_package.training import cross_validate_mlp_with_history
folds, histories, models, norms, feats = cross_validate_mlp_with_history(df, 7, [4,4], ['softplus','softplus'], 80, 1e-3, 8, 'cpu', 42, 'huber')
```

---

## I/O Utilities (`src/my_package/io.py`)

Helpers to manage experiment outputs:

* `make_tag()`
* `create_output_dir(tag)`
* `save_models(models, tag, prefix=None)`
* `save_hyperparams(params, tag, filename=None)`
* `save_norms(norms, tag, filename=None)`

Example:

```python
from my_package.io import make_tag, create_output_dir, save_models, save_hyperparams, save_norms
tag = make_tag()
create_output_dir(tag)
save_models(models, tag)
save_hyperparams(hp, tag)
save_norms(norms, tag)
```

---

## Diagnostics Plotting (`src/my_package/plotting.py`)

`plot_diagnostics(folds, histories, norms, k_folds, epochs, save_path=None, show=True)`

* Loss & R² curves per epoch
* True vs. Predicted scatter for train & val

Example:

```python
from my_package.plotting import plot_diagnostics
plot_diagnostics(folds, histories, norms, k_folds=7, epochs=80, save_path="diag.png")
```

---

## Evaluation (`src/my_package/evaluate.py`)

`evaluate_and_plot_mlp(model_path, norms_json, hidden_dims, activations, test_csv, sample_index=None)`

* Loads model & norms
* Computes R² on test set
* Plots error histogram & true vs. predicted scatter

Example:

```python
from my_package.evaluate import evaluate_and_plot_mlp
evaluate_and_plot_mlp("run1/run1_fold1.pth", "run1/run1_norms.json", [4,4], ['softplus','softplus'], "data/test.csv", sample_index=5)
```

---

## Sample‐Sweep Visualization (`src/my_package/sweep.py`)

`sweep_plot_samples_mlp(model_path, norms_json, hidden_dims, activations, test_csv, sample_index=-1)`

* Sweeps **Input 10** over \[0,3000] for each sample
* Plots model curve + true sample point
* Modes: single, grid, or separate figures

Example:

```python
from my_package.sweep import sweep_plot_samples_mlp
sweep_plot_samples_mlp("run1/run1_fold1.pth", "run1/run1_norms.json", [4,4], ['softplus','softplus'], "data/test.csv", sample_index=-1)
```

---

## Notebooks

Interactive Jupyter notebooks are in `notebook/`:

* **Train.ipynb**
  Executes the full training pipeline:

  1. Data loading & preprocessing
  2. k‐fold training (`cross_validate_mlp_with_history`)
  3. Artifact saving (models, hyperparams, norms)
  4. Diagnostics plotting

* **Test.ipynb**
  Demonstrates test‐set evaluation:

  1. Load model & norms
  2. Compute R²
  3. Error histogram & true vs. predicted scatter (`evaluate_and_plot_mlp`)

* **plot-sweep-samples.ipynb**
  Performs sensitivity analysis on **Input 10**:

  1. Load model & norms
  2. Sweep Input 10 range
  3. Plot response curves with actual sample points (`sweep_plot_samples_mlp`)
