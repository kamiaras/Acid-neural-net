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
