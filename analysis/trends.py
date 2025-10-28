#!/usr/bin/env python3
"""
Script to analyze how model representations evolve across training checkpoints.
Tracks multiple metrics:
- Average distance between datatype 0 and 1 tokens per family
- Normalized distances (relative to overall average distance)
- Linear regression accuracy for datatype classification
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from typing import Dict, List, Tuple
from collections import defaultdict

from analysis.analysis import (
    load_model_and_grammar,
    analyze_datatype_embedding_distances,
    linear_regression_datatype_separation,
    representation_intervention_experiment,
)


def extract_family_distances_from_results(datatype_results: Dict) -> Dict[str, float]:
    """
    Extract normalized family distances from analysis results.

    Args:
        datatype_results: Results from analyze_datatype_embedding_distances
            (which now includes average distance metrics)

    Returns:
        Dictionary containing:
            - 'family_distances': Dict mapping family name to normalized distance
            - 'overall_avg_distance': Average distance between any two tokens
    """
    # Get overall average cosine similarity from datatype_results
    overall_cosine_sim = datatype_results.get(
        "avg_pairwise_cosine_similarity_layer1", 0.0
    )
    overall_avg_distance = 1.0 - overall_cosine_sim

    # Extract similarity matrices from datatype results
    overall_avg_cosine_similarity_dtype0_vs_dtype1 = datatype_results.get(
        "overall_avg_cosine_similarity_dtype0_vs_dtype1", {}
    )

    # Compute average distance per family
    family_distances = {}
    for family, avg_sim in overall_avg_cosine_similarity_dtype0_vs_dtype1.items():
        family_distances[family] = 1.0 - avg_sim

    # Normalize by overall average distance
    normalized_distances = {}
    for family, dist in family_distances.items():
        normalized_distances[family] = dist - overall_avg_distance

    return {
        "family_distances": family_distances,
        "overall_avg_distance": overall_avg_distance,
        "normalized_distances": normalized_distances,
    }


def analyze_checkpoint_evolution(
    run_name: str = "czak7ivo",
    num_sequences: int = 1000,
) -> Dict[str, Dict[int, float]]:
    """
    Analyze how model representations evolve across checkpoints.

    Automatically detects all available checkpoints in the run directory
    and analyzes them in increasing order.

    For each checkpoint, computes:
    - Token embedding distances (overall and per-family)
    - Normalized distances between datatype 0 and 1 tokens
    - Linear regression accuracy for datatype classification

    Args:
        run_name: Name of the run directory
        num_sequences: Number of sequences to use for analysis

    Returns:
        Dictionary containing:
            - 'family_trends': Family-wise normalized distance trends
            - 'overall_distances': Overall average distances per checkpoint
            - 'accuracy_trends': Linear regression accuracy per checkpoint
    """
    # Automatically detect all available checkpoints in the directory
    path = os.path.join("results", "scratch", run_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run directory not found: {path}")

    # Find all checkpoint files matching 'ckpt_{n}.pt' pattern
    checkpoint_pattern = re.compile(r"^ckpt_(\d+)\.pt$")
    checkpoint_steps = []

    for filename in os.listdir(path):
        match = checkpoint_pattern.match(filename)
        if match:
            step = int(match.group(1))
            checkpoint_steps.append(step)

    # Sort checkpoints in increasing order
    checkpoint_steps.sort()

    if len(checkpoint_steps) == 0:
        raise FileNotFoundError(f"No checkpoint files found in {path}")

    print(f"Found {len(checkpoint_steps)} checkpoints: {checkpoint_steps}")

    # Store results across all checkpoints
    family_trends = defaultdict(dict)  # {family: {step: normalized_distance}}
    overall_distances = {}  # {step: overall_avg_distance}
    accuracy_trends = {}  # {step: linear_regression_accuracy}
    intervention_trends = {
        "empty": {},  # {step: percentage}
        "invalid": {},  # {step: percentage}
        "valid": {},  # {step: percentage}
    }

    for step in checkpoint_steps:
        ckpt_name = f"ckpt_{step}.pt"
        print(f"\n{ckpt_name}")

        try:
            # Load model and grammar for this checkpoint
            model, grammar, cfg, dataloader = load_model_and_grammar(
                run_name, ckpt_name
            )

            # Run all analysis functions from analysis.py
            # 1. Distance analysis (includes both average and datatype-specific distances)
            print("  [1/3] Starting distance analysis...")
            datatype_results = analyze_datatype_embedding_distances(
                model,
                dataloader,
                grammar,
                num_sequences=num_sequences,
                showplot=False,
                verbose=False,
                show_progress=True,
            )
            print("  [1/3] Finished distance analysis")

            # 2. Linear regression for datatype separation
            print("  [2/3] Starting linear regression...")
            lr_results = linear_regression_datatype_separation(
                model,
                dataloader,
                grammar,
                max_sequences=num_sequences,
                showplot=False,
                verbose=False,
            )
            print("  [2/3] Finished linear regression")

            # 3. Representation intervention experiments
            print("  [3/3] Starting intervention experiments...")
            intervention_results = representation_intervention_experiment(
                model,
                grammar,
                num_experiments=num_sequences,
                verbose=False,
                show_progress=True,
            )
            print("  [3/3] Finished intervention experiments")

            # Extract and normalize distances by overall average
            distance_results = extract_family_distances_from_results(datatype_results)

            # Compute intervention percentages
            total_interventions = (
                intervention_results["empty_continuations"]
                + intervention_results["invalid_nonempty_continuations"]
                + intervention_results["valid_nonempty_continuations"]
            )
            if total_interventions > 0:
                intervention_trends["empty"][step] = (
                    intervention_results["empty_continuations"]
                    / total_interventions
                    * 100
                )
                intervention_trends["invalid"][step] = (
                    intervention_results["invalid_nonempty_continuations"]
                    / total_interventions
                    * 100
                )
                intervention_trends["valid"][step] = (
                    intervention_results["valid_nonempty_continuations"]
                    / total_interventions
                    * 100
                )

            # Store all metrics for this checkpoint
            overall_distances[step] = distance_results["overall_avg_distance"]
            accuracy_trends[step] = lr_results["accuracy"]
            for family, dist in distance_results["normalized_distances"].items():
                family_trends[family][step] = dist

        except FileNotFoundError:
            print(f"  Error: Checkpoint not found, skipping...")
            continue
        except Exception as e:
            print(f"  Error: {e}")
            continue

    return {
        "family_trends": dict(family_trends),
        "overall_distances": overall_distances,
        "accuracy_trends": accuracy_trends,
        "intervention_trends": intervention_trends,
    }


def plot_evolution_trends(
    family_trends: Dict[str, Dict[int, float]],
    overall_distances: Dict[int, float],
    accuracy_trends: Dict[int, float] = None,
    intervention_trends: Dict[str, Dict[int, float]] = None,
    save_path: str = None,
):
    """
    Create visualization figures showing how model representations evolve across training.

    Generates four separate figures:
    - Figure 1: Normalized distances per family (datatype 0 vs 1, relative to overall average)
    - Figure 2: Raw family distances and overall average distance baseline
    - Figure 3: Linear regression accuracy for datatype classification
    - Figure 4: Intervention experiment percentages (empty, invalid, valid continuations)

    Args:
        family_trends: Dictionary mapping family names to their normalized distance trends
        overall_distances: Dictionary mapping checkpoint steps to overall avg distances
        accuracy_trends: Dictionary mapping checkpoint steps to linear regression accuracy
        intervention_trends: Dictionary with 'empty', 'invalid', 'valid' keys, each mapping steps to percentages
        save_path: Optional base path for saving figures (will append suffixes)
    """
    # Sort checkpoint steps
    steps = sorted(overall_distances.keys())

    # Figure 1: Normalized distances per family
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for family, trend in family_trends.items():
        family_steps = sorted(trend.keys())
        family_values = [trend[s] for s in family_steps]
        ax1.plot(family_steps, family_values, marker="o", label=family, linewidth=2)

    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel(
        "Normalized Distance\n(dtype 0 vs dtype 1) - (overall avg)", fontsize=12
    )
    ax1.set_title(
        "Evolution of Normalized Token Distances Across Training", fontsize=14
    )
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig1.savefig(f"{save_path}_normalized.png", dpi=300, bbox_inches="tight")
        print(f"Saved normalized distances plot to {save_path}_normalized.png")

    plt.show()

    # Figure 2: Raw distances (family-wise dtype 0 vs 1, and overall average)
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Plot overall average distance
    overall_values = [overall_distances[s] for s in steps]
    ax2.plot(
        steps,
        overall_values,
        marker="s",
        label="Overall avg distance",
        linewidth=2.5,
        color="black",
        linestyle="--",
        alpha=0.7,
    )

    # Plot raw family distances (need to reconstruct from normalized)
    for family, trend in family_trends.items():
        family_steps = sorted(trend.keys())
        # Raw distance = normalized_distance + overall_avg_distance
        family_values = [trend[s] + overall_distances[s] for s in family_steps]
        ax2.plot(
            family_steps,
            family_values,
            marker="o",
            label=f"{family} (dtype 0 vs 1)",
            linewidth=2,
            alpha=0.8,
        )

    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Cosine Distance (1 - cosine_similarity)", fontsize=12)
    ax2.set_title("Evolution of Token Distances Across Training", fontsize=14)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig2.savefig(f"{save_path}_raw.png", dpi=300, bbox_inches="tight")
        print(f"Saved raw distances plot to {save_path}_raw.png")

    plt.show()

    # Figure 3: Linear regression accuracy for datatype classification
    if accuracy_trends is not None and len(accuracy_trends) > 0:
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        # Plot accuracy progression across training
        accuracy_steps = sorted(accuracy_trends.keys())
        accuracy_values = [accuracy_trends[s] for s in accuracy_steps]

        ax3.plot(
            accuracy_steps,
            accuracy_values,
            marker="o",
            label="Linear Regression Accuracy",
            linewidth=2.5,
            color="blue",
        )

        # Add reference line at 0.5 (random chance for binary classification)
        ax3.axhline(
            y=0.5,
            color="red",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
            label="Random Chance",
        )

        ax3.set_xlabel("Training Step", fontsize=12)
        ax3.set_ylabel("Accuracy", fontsize=12)
        ax3.set_title(
            "Datatype Classification Accuracy Across Training (Linear Regression)",
            fontsize=14,
        )
        ax3.set_ylim([0, 1.05])  # Accuracy is between 0 and 1
        ax3.legend(loc="best")
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig3.savefig(f"{save_path}_accuracy.png", dpi=300, bbox_inches="tight")
            print(f"Saved accuracy plot to {save_path}_accuracy.png")

        plt.show()

    # Figure 4: Intervention experiment percentages
    if (
        intervention_trends is not None
        and len(intervention_trends.get("empty", {})) > 0
    ):
        fig4, ax4 = plt.subplots(figsize=(10, 6))

        # Plot all three percentages
        intervention_steps = sorted(intervention_trends["empty"].keys())

        empty_values = [intervention_trends["empty"][s] for s in intervention_steps]
        invalid_values = [intervention_trends["invalid"][s] for s in intervention_steps]
        valid_values = [intervention_trends["valid"][s] for s in intervention_steps]

        ax4.plot(
            intervention_steps,
            empty_values,
            marker="o",
            label="Empty continuations",
            linewidth=2.5,
            color="red",
        )
        ax4.plot(
            intervention_steps,
            invalid_values,
            marker="s",
            label="Invalid nonempty continuations",
            linewidth=2.5,
            color="orange",
        )
        ax4.plot(
            intervention_steps,
            valid_values,
            marker="^",
            label="Valid nonempty continuations",
            linewidth=2.5,
            color="green",
        )

        ax4.set_xlabel("Training Step", fontsize=12)
        ax4.set_ylabel("Percentage (%)", fontsize=12)
        ax4.set_title("Intervention Experiment Results Across Training", fontsize=14)
        ax4.set_ylim([0, 105])  # Percentage is between 0 and 100
        ax4.legend(loc="best")
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig4.savefig(f"{save_path}_intervention.png", dpi=300, bbox_inches="tight")
            print(f"Saved intervention plot to {save_path}_intervention.png")

        plt.show()


def main():
    """Main function to run checkpoint evolution analysis"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze evolution of model representations across training checkpoints"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="czak7ivo",
        help="Name of the run directory (default: czak7ivo)",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=1000,
        help="Number of sequences to use for each checkpoint (default: 1000)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Base path for saving plots (default: {run_name}_trends)",
    )

    args = parser.parse_args()

    # Set default save_path to include run name if not specified
    if args.save_path is None:
        args.save_path = f"{args.run_name}_trends"

    print(f"Analyzing checkpoints for run: {args.run_name}")
    print(f"Using {args.num_sequences} sequences per checkpoint")
    print(f"Plots will be saved with prefix: {args.save_path}")
    print("=" * 60)

    # Run comprehensive analysis across all checkpoints
    results = analyze_checkpoint_evolution(
        run_name=args.run_name,
        num_sequences=args.num_sequences,
    )

    # Generate visualization plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    plot_evolution_trends(
        family_trends=results["family_trends"],
        overall_distances=results["overall_distances"],
        accuracy_trends=results["accuracy_trends"],
        intervention_trends=results["intervention_trends"],
        save_path=args.save_path,
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
