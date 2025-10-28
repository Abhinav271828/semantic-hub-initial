"""
Analysis package for model representation analysis and trend visualization.

This package contains:
- analysis.py: Core analysis functions for model representations
- trends.py: Checkpoint trend analysis across training
"""

from analysis.analysis import (
    load_model_and_grammar,
    visualize_outputs_with_logits,
    analyze_datatype_embedding_distances,
    linear_regression_datatype_separation,
    representation_intervention_experiment,
)

from analysis.trends import (
    extract_family_distances_from_results,
    analyze_checkpoint_evolution,
    plot_evolution_trends,
)

__all__ = [
    # From analysis.py
    "load_model_and_grammar",
    "visualize_outputs_with_logits",
    "analyze_datatype_embedding_distances",
    "linear_regression_datatype_separation",
    "representation_intervention_experiment",
    # From trends.py
    "extract_family_distances_from_results",
    "analyze_checkpoint_evolution",
    "plot_evolution_trends",
]
