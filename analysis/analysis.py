#!/usr/bin/env python3
"""
Test script for analyzing model representations and behaviors.
Contains functions for visualization, distance analysis, linear regression, and interventions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pickle as pkl
import os
import random
import argparse
from typing import List, Tuple, Any
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from tqdm import tqdm

from dgp import get_dataloader
from model import GPT


def load_model_and_grammar(
    run_name: str, ckpt_name: str = "latest_ckpt.pt"
) -> Tuple[GPT, Any, Any, Any]:
    """
    Load model, grammar, config, and dataloader from checkpoint.

    Args:
        run_name: Name of the run directory (e.g., 'czak7ivo')
        ckpt_name: Name of checkpoint file (e.g., 'ckpt_100001.pt')

    Returns:
        Tuple of (model, grammar, config, dataloader)
    """
    path = os.path.join("results", "scratch", run_name)

    # Load checkpoint
    state_dict = torch.load(
        os.path.join(path, ckpt_name), map_location="cpu", weights_only=False
    )
    cfg = state_dict["config"]

    # Load grammar
    with open(os.path.join(path, "grammar/PCFG.pkl"), "rb") as f:
        pcfg = pkl.load(f)

    # Create and load model
    model = GPT(cfg.model, pcfg.vocab_size)
    model.load_state_dict(state_dict["net"])
    model.eval()

    # Create dataloader
    dataloader = get_dataloader(
        language=cfg.data.language,
        config=cfg.data.config,
        alpha=cfg.data.alpha,
        prior_type=cfg.data.prior_type,
        num_iters=cfg.data.num_iters * cfg.data.batch_size,
        max_sample_length=cfg.data.max_sample_length,
        seed=cfg.seed,
        batch_size=cfg.data.batch_size,
        num_workers=0,
    )
    try:
        grammar = dataloader.dataset.PCFG
    except:
        grammar = dataloader.dataset.grammar

    return model, grammar, cfg, dataloader


def visualize_outputs_with_logits(
    model: GPT, grammar: Any, cfg: Any, num_samples: int = 5
):
    """
    Generate samples and visualize per-step logits as heatmaps with highlighted chosen tokens.

    Args:
        model: The GPT model
        grammar: The PCFG grammar
        cfg: Model configuration
        num_samples: Number of samples to generate and visualize
    """

    def visualize_logits_table(
        logits_over_time, chosen_token_ids, row_labels, col_labels, cmap="Reds"
    ):
        """Visualize logits as a table with highlighted chosen tokens"""
        V, T = logits_over_time.shape
        fig_w = max(6, min(20, 0.8 * T))
        fig_h = max(6, min(20, 0.22 * V))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        vmin = np.nanmin(logits_over_time)
        vmax = np.nanmax(logits_over_time)
        im = ax.imshow(
            logits_over_time,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xticks(np.arange(T))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(V))
        ax.set_yticklabels(row_labels, fontsize=6)

        # Draw thick borders around chosen token cells
        for t, tok_id in enumerate(chosen_token_ids):
            if 0 <= tok_id < V:
                rect = patches.Rectangle(
                    (t - 0.5, tok_id - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.5,
                )
                ax.add_patch(rect)

        ax.set_xlabel("Generated tokens (time)")
        ax.set_ylabel("Vocabulary tokens")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Logits", rotation=270, labelpad=12)
        fig.tight_layout()
        plt.show()

    def sample_with_logits(
        model, bos_token_id, eos_token_id, max_new_tokens, prune_vocab=None
    ):
        """Sample from model while recording per-step logits"""
        device = next(model.parameters()).device
        inputs = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
        collected_logits = []
        generated_ids = [bos_token_id]

        for _ in range(max_new_tokens):
            logits = model.forward(inputs)[:, -1, :]  # [1, V]
            collected_logits.append(logits.squeeze(0))  # [V]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # [1, 1]
            next_id = next_token.item()
            generated_ids.append(next_id)
            inputs = torch.cat([inputs, next_token], dim=1)
            if next_id == eos_token_id:
                break

        step_logits = (
            torch.stack(collected_logits, dim=0)
            if collected_logits
            else torch.empty(0, model.LM_head.out_features)
        )
        return generated_ids, step_logits

    # Prepare vocabulary labels
    V = grammar.vocab_size
    id_to_token = grammar.id_to_token_map
    row_labels = [id_to_token[i] for i in range(V)]
    bos_id = grammar.vocab["<bos>"]
    eos_id = grammar.vocab["<eos>"]
    max_steps = cfg.data.max_sample_length - 10

    print(f"Generating and visualizing {num_samples} samples...")

    for _ in range(num_samples):
        token_ids, step_logits = sample_with_logits(
            model, bos_id, eos_id, max_new_tokens=max_steps
        )
        gen_token_ids = token_ids[1:]  # exclude BOS for columns
        col_labels = [id_to_token[i] for i in gen_token_ids]
        logits_matrix = step_logits.T.detach().cpu().numpy()  # [V, T]

        visualize_logits_table(
            logits_over_time=logits_matrix,
            chosen_token_ids=gen_token_ids,
            row_labels=row_labels,
            col_labels=col_labels,
            cmap="Reds",
        )


def analyze_datatype_embedding_distances_sequences(
    model: GPT,
    dataloader: Any,
    grammar: Any,
    num_sequences: int = 10_000,
    verbose: bool = True,
    show_progress: bool = True,
    showplot: bool = True,
):
    """Use EOS embedding to compare parallel sequences"""
    layer1_outputs = []

    def _layer1_hook(module, input, output):
        layer1_outputs.append(output.detach().cpu())

    hook_handle = model.transformer.h[0].register_forward_hook(_layer1_hook)

    max_sequence_length = dataloader.dataset.max_sample_length
    bos_token_id = grammar.vocab["<bos>"]
    eos_token_id = grammar.vocab["<eos>"]
    pad_token_id = grammar.vocab["<pad>"]
    sequences_0 = []
    sequences_1 = []
    eos_idxs_0 = []
    eos_idxs_1 = []

    if show_progress:
        iterable = tqdm(
            range(num_sequences), total=num_sequences, desc="Analyzing sequences"
        )
    else:
        iterable = range(num_sequences)

    for _ in iterable:
        sample0, sample1 = grammar.generate_sample(2)

        tokens = grammar.tokenize_sentence(sample0)
        sequence = torch.tensor(
            [bos_token_id]
            + tokens
            + [eos_token_id]
            + [pad_token_id] * (max_sequence_length - len(tokens) - 2)
        )
        eos_idxs_0.append(len(tokens) + 1)
        sequences_0.append(sequence)

        tokens = grammar.tokenize_sentence(sample1)
        sequence = torch.tensor(
            [bos_token_id]
            + grammar.tokenize_sentence(sample1)
            + [eos_token_id]
            + [pad_token_id] * (max_sequence_length - len(tokens) - 2)
        )
        eos_idxs_1.append(len(grammar.tokenize_sentence(sample1)) + 1)
        sequences_1.append(sequence)

    sequences_0 = torch.stack(sequences_0, dim=0)
    sequences_1 = torch.stack(sequences_1, dim=0)

    _ = model(sequences_0)
    all_reps_0 = layer1_outputs.pop(0)
    eos_reps_0 = all_reps_0[torch.arange(num_sequences), eos_idxs_0, :]
    eos_reps_0 = F.normalize(eos_reps_0, dim=-1)
    _ = model(sequences_1)
    all_reps_1 = layer1_outputs.pop(0)
    eos_reps_1 = all_reps_1[torch.arange(num_sequences), eos_idxs_1, :]
    eos_reps_1 = F.normalize(eos_reps_1, dim=-1)

    hook_handle.remove()

    across = torch.mm(eos_reps_0, eos_reps_1.t())
    across_avg = across[torch.triu_indices(num_sequences, num_sequences)].mean()

    within_0 = torch.mm(eos_reps_0, eos_reps_0.t())
    within_1 = torch.mm(eos_reps_1, eos_reps_1.t())
    avg = (
        within_0[torch.triu_indices(num_sequences, num_sequences, offset=1)].mean()
        + within_1[torch.triu_indices(num_sequences, num_sequences, offset=1)].mean()
        + across[torch.triu_indices(num_sequences, num_sequences, offset=0)].mean()
    ) / 3.0

    results = {
        "num_sequences_used": num_sequences,
        "avg_cosine_similarity_layer1": avg,
        "avg_cosine_similarity_dtype0_vs_dtype1": across_avg,
    }

    if verbose:
        pprint(results)
    return results


def analyze_datatype_embedding_distances_tokens(
    model: GPT,
    dataloader: Any,
    grammar: Any,
    num_sequences: int = 1000,
    showplot: bool = True,
    verbose: bool = True,
    show_progress: bool = True,
):
    """
    Analyze distances between embeddings of the same sequence across datatypes.
    Also computes average embedding distances within sequences as a baseline.

    For each sequence:
    1. Get the sequence from the dataset in both datatypes
    2. Pass both sequences separately through the model
    3. Compute similarity matrix M s.t. M[i, j] = sim(token_i-dtype0, token_j-dtype1)
    4. Average similarity matrices across all samples
    5. Also compute average pairwise cosine similarity within each sequence

    Args:
        model: The GPT model
        dataloader: Data loader for getting sequences
        grammar: The PCFG grammar
        num_sequences: Number of sequences to analyze
        showplot: Whether to show plots
        verbose: Whether to print detailed output (default: True)
        show_progress: Whether to show progress bar (default: True)

    Returns:
        Dictionary with analysis results and distance matrices, including:
        - avg_pairwise_cosine_similarity_layer1: Average cosine similarity within sequences
        - avg_embedding_norm_layer1: Average embedding norm
        - overall_avg_cosine_similarity_dtype0_vs_dtype1: Dtype similarities
        - similarity_matrices: Detailed similarity matrices
    """

    # Hook to capture layer 1 outputs
    layer1_outputs = []

    def _layer1_hook(module, input, output):
        layer1_outputs.append(output.detach().cpu())

    hook_handle = model.transformer.h[0].register_forward_hook(_layer1_hook)
    model.eval()

    # Special tokens
    pad_id = grammar.vocab["<pad>"]
    bos_id = grammar.vocab["<bos>"]
    eos_id = grammar.vocab["<eos>"]
    id_to_token = grammar.id_to_token_map

    # Build similarity matrices
    sum_sim_mtx = torch.zeros(grammar.vocab_size, grammar.vocab_size)
    count_sim_mtx = torch.zeros(grammar.vocab_size, grammar.vocab_size)

    # Also compute average distance metrics
    sequence_similarities = []
    total_norm_sum = 0.0
    total_rep_count = 0

    num_sequences_processed = 0

    with torch.no_grad():
        sequence_generator = grammar.sentence_generator(num_sequences)
        if show_progress:
            sequence_generator = tqdm(
                sequence_generator, total=num_sequences, desc="Analyzing sequences"
            )
        for base_sequence in sequence_generator:
            # For example, base_sequence = "bin0 dig1 tern2 dig0 dig2 dig8"
            # Tokenize the same base sequence with both datatypes
            tokens_dtype0 = grammar.tokenize_sentence(base_sequence, dtype=0)
            tokens_dtype1 = grammar.tokenize_sentence(base_sequence, dtype=1)

            # Add BOS and EOS
            seq_dtype0 = torch.tensor(
                [bos_id] + tokens_dtype0 + [eos_id], dtype=torch.long
            ).unsqueeze(0)
            seq_dtype1 = torch.tensor(
                [bos_id] + tokens_dtype1 + [eos_id], dtype=torch.long
            ).unsqueeze(0)

            # Pass both sequences through the model
            _ = model(seq_dtype0)
            reps_dtype0 = layer1_outputs.pop(0)  # [1, L, C]

            _ = model(seq_dtype1)
            reps_dtype1 = layer1_outputs.pop(0)  # [1, L, C]

            # Extract token ids (excluding BOS and EOS)
            reps0 = reps_dtype0[0, 1:-1, :]  # [L-2, C]
            reps1 = reps_dtype1[0, 1:-1, :]  # [L-2, C]
            reps0_norm = F.normalize(reps0, dim=1)  # [N, C]
            reps1_norm = F.normalize(reps1, dim=1)  # [N, C]

            # Average pairwise similarity across datatypes
            seq_sim_matrix = torch.mm(reps0_norm, reps1_norm.t())  # [N, N]
            for i_pos, i_tok in enumerate(seq_dtype0[0, 1:-1].tolist()):
                for j_pos, j_tok in enumerate(seq_dtype1[0, 1:-1].tolist()):
                    sum_sim_mtx[i_tok, j_tok] += seq_sim_matrix[i_pos, j_pos]
                    count_sim_mtx[i_tok, j_tok] += 1

            # Compute average pairwise similarity within each sequence for baseline
            if reps0.size(0) > 1:
                # Compute cosine similarity matrix for dtype0 tokens
                cosine_sim_within = torch.mm(reps0_norm, reps0_norm.t())  # [N, N]
                # Extract upper triangle (excluding diagonal)
                N = cosine_sim_within.size(0)
                triu_indices = torch.triu_indices(N, N, offset=1)
                upper_triangle_sims = cosine_sim_within[
                    triu_indices[0], triu_indices[1]
                ]
                # Store mean similarity for this sequence
                sequence_similarities.append(upper_triangle_sims.mean().item())
            if reps1.size(0) > 1:
                # Compute cosine similarity matrix for dtype0 tokens
                cosine_sim_within = torch.mm(reps1_norm, reps1_norm.t())  # [N, N]
                # Extract upper triangle (excluding diagonal)
                N = cosine_sim_within.size(0)
                triu_indices = torch.triu_indices(N, N, offset=1)
                upper_triangle_sims = cosine_sim_within[
                    triu_indices[0], triu_indices[1]
                ]
                # Store mean similarity for this sequence
                sequence_similarities.append(upper_triangle_sims.mean().item())

            # Accumulate norms
            norms = torch.norm(reps0, dim=1)  # [N]
            total_norm_sum += norms.sum().item()
            total_rep_count += norms.numel()

            num_sequences_processed += 1

    hook_handle.remove()

    # Compute final avg sim mtx
    sim_mtx = sum_sim_mtx / count_sim_mtx

    # Compute average cosine similarity and embedding norm
    if len(sequence_similarities) > 0:
        avg_cosine_similarity = sum(sequence_similarities) / len(sequence_similarities)
    else:
        avg_cosine_similarity = float("nan")

    avg_embedding_norm_layer1 = (
        (total_norm_sum / total_rep_count) if total_rep_count > 0 else float("nan")
    )

    if showplot:
        mat = sim_mtx
        fig_w = max(4, min(14, 0.35 * mat.shape[1]))
        fig_h = max(4, min(14, 0.25 * mat.shape[0]))
        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(mat, aspect="auto", cmap="viridis")
        plt.colorbar(label=f"Cosine similarity: dtype-0 vs dtype-1")
        plt.yticks(
            np.arange(mat.shape[0]),
            [f"{id_to_token[i]}" for i in range(mat.shape[0])],
            fontsize=6,
        )
        plt.xticks(
            np.arange(mat.shape[1]),
            [f"{id_to_token[j]}" for j in range(mat.shape[1])],
            fontsize=6,
            rotation=90,
        )
        plt.title(f"Cosine similarity heatmap (dtype 0 vs dtype 1)")
        plt.tight_layout()
        plt.show()

    results = {
        "num_sequences_used": num_sequences_processed,
        "num_sequences_with_valid_tokens": len(sequence_similarities),
        "avg_cosine_similarity_layer1": avg_cosine_similarity,
        "avg_embedding_norm_layer1": avg_embedding_norm_layer1,
        "num_token_pairs": count_sim_mtx,
        "avg_cosine_similarity_dtype0_vs_dtype1": sim_mtx[~sim_mtx.isnan()].mean(),
        "similarity_matrices": sim_mtx,
    }

    if verbose:
        pprint(results)
    return results


def linear_regression_datatype_separation_tokens(
    model: GPT,
    dataloader: Any,
    grammar: Any,
    max_sequences: int = 1200,
    showplot: bool = True,
    verbose: bool = True,
):
    """
    Learn a linear regression to classify token embeddings according to their datatype.

    Args:
        model: The GPT model
        dataloader: Data loader for getting sequences
        grammar: The PCFG grammar
        max_sequences: Maximum number of sequences to use
        showplot: Whether to display the PCA visualization plot
        verbose: Whether to print detailed output (default: True)

    Returns:
        Dictionary with classification results
    """

    def collect_layer1_embeddings_with_dtype(
        model, dataloader, grammar, max_sequences=1200, exclude_special=True
    ):
        """Collect layer-1 embeddings with dtype and family labels"""
        buf = []

        def _hook(module, inp, out):
            buf.append(out.detach().cpu())

        handle = model.transformer.h[0].register_forward_hook(_hook)

        pad_id = grammar.vocab["<pad>"]
        bos_id = grammar.vocab["<bos>"]
        eos_id = grammar.vocab["<eos>"]
        id_to_token = grammar.id_to_token_map

        reps: List[torch.Tensor] = []
        dtypes: List[int] = []
        seen = 0

        with torch.no_grad():
            for batch in dataloader:
                seqs, _, _ = batch
                B, L = seqs.size()
                _ = model(seqs)
                x1 = buf.pop(0)  # [B, L, C]

                seq_ids = seqs.detach().cpu()
                mask = seq_ids != pad_id
                if exclude_special:
                    mask &= (seq_ids != bos_id) & (seq_ids != eos_id)

                idx = mask.nonzero(as_tuple=False)
                if idx.numel() > 0:
                    b_idx = idx[:, 0]
                    t_idx = idx[:, 1]
                    tok_ids = seq_ids[b_idx, t_idx]
                    tok_reps = x1[b_idx, t_idx, :]

                    # Parse dtype and family from token string
                    for rep, tid in zip(tok_reps, tok_ids):
                        name = id_to_token[int(tid)]
                        if "-" not in name:
                            continue
                        base, dtype_str = name.rsplit("-", 1)
                        try:
                            dtype = int(dtype_str)
                        except ValueError:
                            continue
                        # Extract family prefix (letters at start of base)
                        fam = "".join([ch for ch in base if ch.isalpha()]) or "other"
                        reps.append(rep)
                        dtypes.append(dtype)

                seen += B
                if seen >= max_sequences:
                    break

        handle.remove()
        if len(reps) == 0:
            return (
                torch.empty(0, model.transformer.h[0].n_embd),
                torch.empty(0, dtype=torch.long),
                [],
            )
        X = torch.stack(reps, dim=0)  # [N, C]
        y = torch.tensor(dtypes, dtype=torch.float32)  # [N]
        return X, y

    # Gather dataset
    X, y = collect_layer1_embeddings_with_dtype(
        model, dataloader, grammar, max_sequences=max_sequences
    )
    N, C = X.size()
    if verbose:
        print({"num_embeddings_collected": int(N), "embedding_dim": int(C)})

    # Train/test split
    perm = torch.randperm(N)
    train_ratio = 0.8
    n_train = int(train_ratio * N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Fit linear regression (least squares) with bias term
    X_train_aug = torch.cat(
        [X_train, torch.ones(X_train.size(0), 1)], dim=1
    )  # [N, C+1]
    solution = torch.linalg.lstsq(X_train_aug, y_train).solution  # [C+1]
    w, b = solution[:-1], solution[-1]

    # Predict with threshold 0.5
    with torch.no_grad():
        y_logits = X_test @ w + b
        y_pred = (y_logits >= 0.5).float()
        y_true = y_test

    # Metrics: accuracy, precision, recall (positive class = dtype 1)
    TP = ((y_pred == 1) & (y_true == 1)).sum().item()
    TN = ((y_pred == 0) & (y_true == 0)).sum().item()
    FP = ((y_pred == 1) & (y_true == 0)).sum().item()
    FN = ((y_pred == 0) & (y_true == 1)).sum().item()

    accuracy = (TP + TN) / max(1, len(y_true))
    precision = TP / max(1, (TP + FP))
    recall = TP / max(1, (TP + FN))

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
    if verbose:
        print(results)

    # Visualization: project to 2D (PCA on train set), plot points and decision boundary
    if showplot:
        M_vis = min(1000, X_train.size(0))
        X_train_sample = X_train[:M_vis]
        # PCA via SVD
        Xc = X_train_sample - X_train_sample.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        P2 = Vh[:2].T  # [C, 2]
        mu = X_train_sample.mean(dim=0)

        def project_2d(X_in):
            return (X_in - mu) @ P2  # [N, 2]

        X2_train = project_2d(X_train[:300])
        y2_train = y_train[:300]
        X2_test = project_2d(X_test[:300])
        y2_test = y_test[:300]

        # Project linear boundary: w2d^T y + const = 0 where y = (x - mu) @ P2
        w2d = P2.T @ w
        const = w @ mu + b - 0.5

        marker_map = {0.0: "o", 1.0: "x"}

        plt.figure(figsize=(7, 6))
        # Plot train
        for i in range(len(y2_train)):
            marker = marker_map.get(float(y2_train[i].item()), "o")
            plt.scatter(
                X2_train[i, 0], X2_train[i, 1], c=["red"], marker=marker, alpha=0.7
            )
        # Plot test
        for i in range(len(y2_test)):
            marker = marker_map.get(float(y2_test[i].item()), "o")
            plt.scatter(
                X2_test[i, 0],
                X2_test[i, 1],
                c=["red"],
                marker=marker,
                alpha=0.9,
                edgecolor="k",
                linewidths=0.3,
            )

        # Decision boundary line: w2d_x * x + w2d_y * y + const = 0
        x_min = min(X2_train[:, 0].min().item(), X2_test[:, 0].min().item())
        x_max = max(X2_train[:, 0].max().item(), X2_test[:, 0].max().item())
        xx = np.linspace(x_min, x_max, 200)
        if abs(w2d[1].item()) > 1e-8:
            yy = -(w2d[0].item() * xx + const.item()) / w2d[1].item()
            plt.plot(xx, yy, "k--", label="linear boundary (proj)")

        dtype_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_map[d],
                color="k",
                label=f"dtype {int(d)}",
                linestyle="None",
                markersize=8,
            )
            for d in [0.0, 1.0]
        ]

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(
            "Datatype linear regression in 2D PCA space (color=family, marker=dtype)"
        )
        plt.legend(handles=dtype_handles, frameon=True)
        plt.tight_layout()
        plt.show()

    return results


def linear_regression_datatype_separation_sequences(
    model: GPT,
    dataloader: Any,
    grammar: Any,
    max_sequences: int = 1200,
    showplot: bool = True,
    verbose: bool = True,
):
    """
    Learn a linear regression to classify sequences embeddings according to their datatype.

    Args:
        model: The GPT model
        dataloader: Data loader for getting sequences
        grammar: The PCFG grammar
        max_sequences: Maximum number of sequences to use
        showplot: Whether to display the PCA visualization plot
        verbose: Whether to print detailed output (default: True)

    Returns:
        Dictionary with classification results
    """

    def collect_layer1_embeddings_with_dtype(
        model, dataloader, grammar, max_sequences=1200, exclude_special=True
    ):
        """Collect layer-1 embeddings with dtype and family labels"""
        buf = []

        def _hook(module, inp, out):
            buf.append(out.detach().cpu())

        handle = model.transformer.h[0].register_forward_hook(_hook)

        reps: List[torch.Tensor] = []
        dtypes: List[int] = []
        seen = 0

        with torch.no_grad():
            for batch in dataloader:
                seqs, lens, dts = batch
                B, _ = seqs.size()
                _ = model(seqs)
                x1 = buf.pop(0)  # [B, L, C]
                seq_reps = x1[torch.arange(x1.size(0)), lens.long() + 1, :]  # eos reps
                reps += seq_reps
                dtypes += dts

                seen += B
                if seen >= max_sequences:
                    break

        handle.remove()
        if len(reps) == 0:
            return (
                torch.empty(0, model.transformer.h[0].n_embd),
                torch.empty(0, dtype=torch.long),
                [],
            )
        X = torch.stack(reps, dim=0)  # [N, C]
        y = torch.tensor(dtypes, dtype=torch.float32)  # [N]
        return X, y

    # Gather dataset
    X, y = collect_layer1_embeddings_with_dtype(
        model, dataloader, grammar, max_sequences=max_sequences
    )
    N, C = X.size()
    if verbose:
        print({"num_embeddings_collected": int(N), "embedding_dim": int(C)})

    # Train/test split
    perm = torch.randperm(N)
    train_ratio = 0.8
    n_train = int(train_ratio * N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Fit linear regression (least squares) with bias term
    X_train_aug = torch.cat(
        [X_train, torch.ones(X_train.size(0), 1)], dim=1
    )  # [N, C+1]
    solution = torch.linalg.lstsq(X_train_aug, y_train).solution  # [C+1]
    w, b = solution[:-1], solution[-1]

    # Predict with threshold 0.5
    with torch.no_grad():
        y_logits = X_test @ w + b
        y_pred = (y_logits >= 0.5).float()
        y_true = y_test

    # Metrics: accuracy, precision, recall (positive class = dtype 1)
    TP = ((y_pred == 1) & (y_true == 1)).sum().item()
    TN = ((y_pred == 0) & (y_true == 0)).sum().item()
    FP = ((y_pred == 1) & (y_true == 0)).sum().item()
    FN = ((y_pred == 0) & (y_true == 1)).sum().item()

    accuracy = (TP + TN) / max(1, len(y_true))
    precision = TP / max(1, (TP + FP))
    recall = TP / max(1, (TP + FN))

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
    if verbose:
        print(results)

    # Visualization: project to 2D (PCA on train set), plot points and decision boundary
    if showplot:
        M_vis = min(1000, X_train.size(0))
        X_train_sample = X_train[:M_vis]
        # PCA via SVD
        Xc = X_train_sample - X_train_sample.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        P2 = Vh[:2].T  # [C, 2]
        mu = X_train_sample.mean(dim=0)

        def project_2d(X_in):
            return (X_in - mu) @ P2  # [N, 2]

        X2_train = project_2d(X_train[:300])
        y2_train = y_train[:300]
        X2_test = project_2d(X_test[:300])
        y2_test = y_test[:300]

        # Project linear boundary: w2d^T y + const = 0 where y = (x - mu) @ P2
        w2d = P2.T @ w
        const = w @ mu + b - 0.5

        # Assign a color per family and marker per dtype
        marker_map = {0.0: "o", 1.0: "x"}

        plt.figure(figsize=(7, 6))
        # Plot train
        for i in range(len(y2_train)):
            marker = marker_map.get(float(y2_train[i].item()), "o")
            plt.scatter(
                X2_train[i, 0], X2_train[i, 1], c=["red"], marker=marker, alpha=0.7
            )
        # Plot test
        for i in range(len(y2_test)):
            marker = marker_map.get(float(y2_test[i].item()), "o")
            plt.scatter(
                X2_test[i, 0],
                X2_test[i, 1],
                c=["red"],
                marker=marker,
                alpha=0.9,
                edgecolor="k",
                linewidths=0.3,
            )

        # Decision boundary line: w2d_x * x + w2d_y * y + const = 0
        x_min = min(X2_train[:, 0].min().item(), X2_test[:, 0].min().item())
        x_max = max(X2_train[:, 0].max().item(), X2_test[:, 0].max().item())
        xx = np.linspace(x_min, x_max, 200)
        if abs(w2d[1].item()) > 1e-8:
            yy = -(w2d[0].item() * xx + const.item()) / w2d[1].item()
            plt.plot(xx, yy, "k--", label="linear boundary (proj)")

        dtype_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_map[d],
                color="k",
                label=f"dtype {int(d)}",
                linestyle="None",
                markersize=8,
            )
            for d in [0.0, 1.0]
        ]

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(
            "Datatype linear regression in 2D PCA space (color=family, marker=dtype)"
        )
        plt.legend(handles=dtype_handles, frameon=True)
        plt.tight_layout()
        plt.show()

    return results


def representation_intervention_experiment(
    model: GPT,
    grammar: Any,
    num_experiments: int = 100,
    verbose: bool = True,
    show_progress: bool = True,
):
    """
    Run representation swapping experiment to test if swapping datatype representations affects generation.

    Args:
        model: The GPT model
        grammar: The PCFG grammar
        num_experiments: Number of experiments to run
        verbose: Whether to print detailed output (default: True)
        show_progress: Whether to show progress bar (default: True)

    Returns:
        Dictionary with experiment results
    """

    class LayerHook:
        def __init__(self, layer_idx, replacement_rep=None):
            self.layer_idx = layer_idx
            self.replacement_rep = replacement_rep
            self.captured_rep = None
            self.hook_handle = None

        def hook_fn(self, module, input, output):
            if self.replacement_rep is not None:
                # Replace the representation at the specified position
                output = output.clone()
                output[0, self.swap_position, :] = self.replacement_rep
            else:
                # Capture the representation
                self.captured_rep = output.clone()
            return output

        def register(self, model, swap_position):
            self.swap_position = swap_position
            self.hook_handle = model.transformer.h[
                self.layer_idx
            ].register_forward_hook(self.hook_fn)

        def remove(self):
            if self.hook_handle:
                self.hook_handle.remove()

    def generate_sample_with_dtype(grammar, base_sequence, dtype):
        """Tokenize the same base sequence with a specific datatype"""
        bos_id = grammar.vocab["<bos>"]
        eos_id = grammar.vocab["<eos>"]

        # Tokenize the same base sequence with the specified datatype
        token_ids = grammar.tokenize_sentence(base_sequence, dtype)

        # Add BOS and EOS
        full_sequence = [bos_id] + token_ids + [eos_id]
        return torch.tensor(full_sequence, dtype=torch.long)

    def continue_generation(model, input_sequence, max_new_tokens=10):
        """Continue generation from a given sequence"""
        device = next(model.parameters()).device
        current_seq = input_sequence.clone().to(device)

        generated_tokens = []
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(current_seq)[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            if next_token.item() == grammar.vocab["<eos>"]:
                break

            generated_tokens.append(next_token.item())
            current_seq = torch.cat([current_seq, next_token], dim=1)

        return generated_tokens

    def run_representation_swap_experiment_corrected(
        model, grammar, num_experiments=100, show_progress=True
    ):
        """Run the representation swapping experiment with the same base sample"""
        results = {
            "empty_continuations": 0,
            "invalid_nonempty_continuations": 0,
            "valid_nonempty_continuations": 0,
            "experiment_details": [],
        }

        experiment_range = range(num_experiments)
        if show_progress:
            experiment_range = tqdm(experiment_range, desc="Running interventions")

        for exp_idx in experiment_range:
            # Generate the SAME base sample and tokenize with both datatypes
            base_sequence = next(grammar.sentence_generator(1))
            sample0 = generate_sample_with_dtype(grammar, base_sequence, 0)
            sample1 = generate_sample_with_dtype(grammar, base_sequence, 1)

            # Ensure we have at least 2 tokens after the swap position (including <eos>)
            # Need: BOS + ... + swap_pos + at least 2 more tokens
            if len(sample0) < 4 or len(sample1) < 4:
                continue

            # Pick a random token position ensuring at least 2 tokens after (including EOS)
            max_swap_pos = (
                min(len(sample0), len(sample1)) - 3
            )  # -3 to ensure at least 2 tokens after swap position
            if max_swap_pos <= 0:
                continue

            swap_position = random.randint(1, max_swap_pos)

            # Pass sample up to swap position in dtype0, capture representation
            prefix0 = sample0[: swap_position + 1].unsqueeze(
                0
            )  # Include the token at swap_position

            # Capture representation from dtype0
            hook0 = LayerHook(layer_idx=0)  # Use layer 0 (first transformer block)
            hook0.register(model, swap_position)

            with torch.no_grad():
                _ = model(prefix0)

            captured_rep = hook0.captured_rep[0, swap_position, :].clone()
            hook0.remove()

            # Now pass prefix0 through model with dtype1's representation at swap_position
            hook1 = LayerHook(layer_idx=0, replacement_rep=captured_rep)
            hook1.register(model, swap_position)

            with torch.no_grad():
                _ = model(prefix0)

            hook1.remove()

            # Continue generation with the intervention
            continuation = continue_generation(model, prefix0, max_new_tokens=5)

            # Detokenize continuation for human-readable output
            continuation_detokenized = ""
            if len(continuation) > 0:
                continuation_tensor = torch.tensor(continuation, dtype=torch.long)
                continuation_detokenized = grammar.detokenize_sentence(
                    continuation_tensor.numpy()
                )
                # Remove any BOS/EOS markers if present
                continuation_detokenized = (
                    continuation_detokenized.replace("<bos>", "")
                    .replace("<eos>", "")
                    .strip()
                )

            # Check if continuation is empty
            if len(continuation) == 0:
                results["empty_continuations"] += 1
                is_valid = None
            else:
                # Check if continuation is valid by checking grammaticality
                # Reconstruct full sequence: prefix + continuation
                full_sequence_tokens = prefix0[0].tolist() + continuation

                # Detokenize the sequence
                full_sequence_tensor = torch.tensor(
                    full_sequence_tokens, dtype=torch.long
                )
                detokenized_sentence = grammar.detokenize_sentence(
                    full_sequence_tensor.numpy()
                )

                # Extract the sentence (remove BOS/EOS markers)
                sentence = (
                    detokenized_sentence.split("<eos>")[0].split("<bos>")[1].strip()
                )

                # Check grammaticality
                grammaticality, _ = grammar.check_grammaticality(sentence)
                is_valid = grammaticality[0]

                if is_valid:
                    results["valid_nonempty_continuations"] += 1
                else:
                    results["invalid_nonempty_continuations"] += 1

            # Store experiment details
            results["experiment_details"].append(
                {
                    "exp_idx": exp_idx,
                    "swap_position": swap_position,
                    "prefix_length": len(prefix0[0]),
                    "continuation": continuation,
                    "continuation_detokenized": continuation_detokenized,
                    "is_empty": len(continuation) == 0,
                    "is_valid": is_valid,
                    "base_sequence": base_sequence,
                    "sample0_tokens": sample0.tolist(),
                    "sample1_tokens": sample1.tolist(),
                }
            )

        return results

    # Run the corrected experiment
    if verbose:
        print("Starting representation swap experiment...")
    experiment_results = run_representation_swap_experiment_corrected(
        model, grammar, num_experiments=num_experiments, show_progress=show_progress
    )

    # Print results
    total = (
        experiment_results["empty_continuations"]
        + experiment_results["invalid_nonempty_continuations"]
        + experiment_results["valid_nonempty_continuations"]
    )
    if verbose:
        print(f"\nExperiment Results:")
        print(f"Total experiments: {total}")
        print(f"(i) Empty continuations: {experiment_results['empty_continuations']}")
        print(
            f"(ii) Invalid nonempty continuations: {experiment_results['invalid_nonempty_continuations']}"
        )
        print(
            f"(iii) Valid nonempty continuations: {experiment_results['valid_nonempty_continuations']}"
        )
        if total > 0:
            print(f"\nPercentages:")
            print(
                f"  Empty: {experiment_results['empty_continuations'] / total * 100:.1f}%"
            )
            print(
                f"  Invalid nonempty: {experiment_results['invalid_nonempty_continuations'] / total * 100:.1f}%"
            )
            print(
                f"  Valid nonempty: {experiment_results['valid_nonempty_continuations'] / total * 100:.1f}%"
            )

        # Show a few examples
        print(f"\nFirst 5 experiment details:")
        for i, detail in enumerate(experiment_results["experiment_details"][:5]):
            print(f"Experiment {i+1}:")

            # Format base sequence with intervened token in red
            base_tokens = detail["base_sequence"].split()
            swap_pos = detail["swap_position"]
            # swap_position is 1-indexed after BOS, so token index is swap_position - 1
            token_idx = swap_pos - 1

            # Build formatted base sequence with ANSI color codes
            formatted_tokens = []
            for idx, token in enumerate(base_tokens):
                if idx == token_idx:
                    # Red color for intervened token
                    formatted_tokens.append(f"\033[91m{token}\033[0m")
                else:
                    formatted_tokens.append(token)
            formatted_base_sequence = " ".join(formatted_tokens)

            print(f"  Base sequence: {formatted_base_sequence}")
            print(f"  Swap position: {swap_pos}")
            print(f"  Continuation (detokenized): {detail['continuation_detokenized']}")
            print(f"  Is empty: {detail['is_empty']}")
            print(f"  Is valid: {detail['is_valid']}")
            print()

    return experiment_results


def main():
    """Main function to run all analyses"""
    parser = argparse.ArgumentParser(
        description="Test script for analyzing model representations and behaviors"
    )

    # Model arguments
    parser.add_argument(
        "--run_name",
        type=str,
        default="czak7ivo",
        help="Name of the run directory (default: czak7ivo)",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="ckpt_100001.pt",
        help="Name of checkpoint file (default: ckpt_100001.pt)",
    )

    # Analysis selection flags
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Run visualization of outputs with logits",
    )
    parser.add_argument(
        "--distance",
        action="store_true",
        help="Run distance analysis (average distances and datatype-specific distances, per token or sequence)",
    )
    parser.add_argument(
        "--linear_regression",
        action="store_true",
        help="Run linear regression for datatype separation",
    )
    parser.add_argument(
        "--intervention",
        action="store_true",
        help="Run representation intervention experiment",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all analyses (default if no specific analysis is selected)",
    )

    # Parameters for analyses
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples for visualization (default: 3)",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=1000,
        help="Number of sequences for distance analyses (default: 1000)",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=1200,
        help="Maximum sequences for linear regression (default: 1200)",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=100,
        help="Number of experiments for intervention (default: 100)",
    )

    args = parser.parse_args()

    # If no specific analysis is selected, run all
    if not any(
        [
            args.visualize,
            args.distance,
            args.linear_regression,
            args.intervention,
        ]
    ):
        args.all = True

    # Load model and grammar
    print(f"Loading model from {args.run_name}/{args.ckpt_name}...")
    model, grammar, cfg, dataloader = load_model_and_grammar(
        args.run_name, args.ckpt_name
    )

    print("Model loaded successfully!")
    print(f"Vocabulary size: {grammar.vocab_size}")
    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")

    # Run selected analyses
    if args.all or args.visualize:
        print("\n" + "=" * 50)
        print("1. VISUALIZING OUTPUTS WITH LOGITS")
        print("=" * 50)
        visualize_outputs_with_logits(model, grammar, cfg, num_samples=args.num_samples)

    if args.all or args.distance:
        print("\n" + "=" * 50)
        print("2. DISTANCE ANALYSIS (AVERAGE & DATATYPE-SPECIFIC))")
        print("=" * 50)
        match cfg.data.unit:
            case "tok":
                analyze_datatype_embedding_distances_tokens(
                    model, dataloader, grammar, num_sequences=args.num_sequences
                )
            case "seq":
                analyze_datatype_embedding_distances_sequences(
                    model, dataloader, grammar, num_sequences=args.num_sequences
                )

    if args.all or args.linear_regression:
        print("\n" + "=" * 50)
        print("3. LINEAR REGRESSION SEPARATING DATATYPES")
        print("=" * 50)
        match cfg.data.unit:
            case "tok":
                linear_regression_datatype_separation_tokens(
                    model, dataloader, grammar, max_sequences=args.max_sequences
                )
            case "seq":
                linear_regression_datatype_separation_sequences(
                    model, dataloader, grammar, max_sequences=args.max_sequences
                )

    if (args.all or args.intervention) and (cfg.data.unit == "tok"):
        print("\n" + "=" * 50)
        print("4. REPRESENTATION INTERVENTION EXPERIMENT")
        print("=" * 50)
        representation_intervention_experiment(
            model, grammar, num_experiments=args.num_experiments
        )


if __name__ == "__main__":
    main()
