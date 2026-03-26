"""
main.py
=======
4D Seismic Hardening / Softening — Semantic Segmentation
---------------------------------------------------------
Entry point that orchestrates:
  1. Data loading & preprocessing
  2. Single-architecture training  OR  full comparison of all three
  3. Rich visualisation of results (training curves, IoU, GT vs Pred, diff map)

Usage examples
--------------
# Train a single architecture
python main.py --architecture unet_deep

# Compare all three architectures
python main.py --compare

# Override any default
python main.py --compare --epochs 30 --batch-size 8 --sigma 0.4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from src.model import (
    ARCH_NAMES,
    DataConfig,
    TrainingConfig,
    compare_architectures,
    load_data,
    preprocess_for_segmentation,
    train_and_evaluate,
)

# ── Shared colourmaps ─────────────────────────────────────────────────────────
SEISMIC_RWB = ListedColormap(["red", "white", "blue"])
SEG_CMAP    = ListedColormap(["red", "white", "blue"])
SEG_NORM    = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], SEG_CMAP.N)
CLASS_NAMES = ["Softening (0)", "Neutral (1)", "Hardening (2)"]


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _seismic_imshow(
    ax,
    data:        np.ndarray,
    title:       str,
    xlabel:      str,
    ylabel:      str,
    cbar_label:  str,
) -> None:
    """Single-panel imshow — Red/White/Blue, ±3σ norm."""
    norm = plt.Normalize(vmin=-np.std(data) * 3, vmax=np.std(data) * 3)
    im   = ax.imshow(
        data, aspect="auto", cmap=SEISMIC_RWB, norm=norm,
        extent=[0, data.shape[1], data.shape[0], 0],
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


def plot_segmentation_sample(
    x_sample: np.ndarray,
    y_true:   np.ndarray,
    y_pred:   np.ndarray | None = None,
    title:    str = "Semantic segmentation",
    out_path: str | None = None,
) -> None:
    """
    4-panel figure (or 3-panel when y_pred is None):
      1. Input dIP  (Red/White/Blue, ±3σ)
      2. Ground-truth mask
      3. Predicted mask  [optional]
      4. Diff map — green=correct, red=wrong  [optional, only if y_pred given]
    """
    mask_patches = [
        mpatches.Patch(color="red",       label="0 — Softening"),
        mpatches.Patch(color="white",     label="1 — Neutral"),
        mpatches.Patch(color="royalblue", label="2 — Hardening"),
    ]

    ncols = 4 if y_pred is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    fig.patch.set_facecolor("white")

    # Panel 1 — Input dIP
    raw  = x_sample.squeeze()
    norm = plt.Normalize(vmin=-np.std(raw) * 3, vmax=np.std(raw) * 3)
    im0  = axes[0].imshow(
        raw, aspect="auto", cmap=SEISMIC_RWB, norm=norm,
        extent=[0, raw.shape[1], raw.shape[0], 0], interpolation="nearest",
    )
    plt.colorbar(im0, ax=axes[0], label="dIP")
    axes[0].set_title("Input dIP", fontsize=10)
    axes[0].set_xlabel("Crossline (W)", fontsize=8)
    axes[0].set_ylabel("Time (H)", fontsize=8)

    # Panel 2 — Ground-truth mask
    axes[1].imshow(
        y_true, aspect="auto", cmap=SEG_CMAP, norm=SEG_NORM,
        extent=[0, y_true.shape[1], y_true.shape[0], 0], interpolation="nearest",
    )
    axes[1].legend(handles=mask_patches, loc="lower right", fontsize=7)
    axes[1].set_title("Ground-truth mask\n(0=soft · 1=neutral · 2=hard)", fontsize=10)
    axes[1].set_xlabel("Crossline (W)", fontsize=8)
    axes[1].set_ylabel("Time (H)", fontsize=8)

    if y_pred is not None:
        # Panel 3 — Predicted mask
        axes[2].imshow(
            y_pred, aspect="auto", cmap=SEG_CMAP, norm=SEG_NORM,
            extent=[0, y_pred.shape[1], y_pred.shape[0], 0], interpolation="nearest",
        )
        axes[2].legend(handles=mask_patches, loc="lower right", fontsize=7)
        axes[2].set_title("Predicted mask", fontsize=10)
        axes[2].set_xlabel("Crossline (W)", fontsize=8)
        axes[2].set_ylabel("Time (H)", fontsize=8)

        # Panel 4 — Diff map
        correct   = (y_true == y_pred).astype(np.uint8)
        pct_ok    = correct.mean() * 100
        diff_cmap = ListedColormap(["red", "limegreen"])
        axes[3].imshow(
            correct, aspect="auto", cmap=diff_cmap, vmin=0, vmax=1,
            extent=[0, correct.shape[1], correct.shape[0], 0], interpolation="nearest",
        )
        diff_patches = [
            mpatches.Patch(color="limegreen", label="Correct"),
            mpatches.Patch(color="red",       label="Wrong"),
        ]
        axes[3].legend(handles=diff_patches, loc="lower right", fontsize=7)
        axes[3].set_title(f"Diff map\n{pct_ok:.1f}% correct", fontsize=10)
        axes[3].set_xlabel("Crossline (W)", fontsize=8)
        axes[3].set_ylabel("Time (H)", fontsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved → {out_path}")
    plt.close(fig)


def plot_training_curves(
    history,
    arch:     str,
    out_path: str | None = None,
) -> None:
    """Loss, accuracy, and Dice curves for a single architecture."""
    hist      = history.history
    epochs    = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for ax, train_key, val_key, ylabel in [
        (axes[0], "loss",      "val_loss",      "Loss"),
        (axes[1], "accuracy",  "val_accuracy",  "Accuracy"),
        (axes[2], "dice_coef", "val_dice_coef", "Dice"),
    ]:
        ax.plot(epochs, hist[train_key], lw=2, color="steelblue", label="train")
        ax.plot(epochs, hist[val_key],   lw=2, color="tomato", ls="--", label="val")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title("Combined Loss (CE + Dice)")
    axes[1].set_title("Pixel Accuracy")
    axes[2].set_title("Dice Coefficient")
    fig.suptitle(f"Training curves — {arch}", fontsize=13)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
    plt.close(fig)


def plot_iou_bars(
    metrics:  dict,
    arch:     str,
    out_path: str | None = None,
) -> None:
    """Per-class IoU bar chart for a single architecture."""
    iou_vals   = [metrics[f"class_{c}_iou"] for c in range(3)]
    iou_colors = ["red", "white",     "royalblue"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASS_NAMES, iou_vals, color=iou_colors, edgecolor="k")
    ax.axhline(metrics["mean_iou"], color="black", lw=1.5, ls="--",
               label=f"mIoU = {metrics['mean_iou']:.3f}")
    for bar, val in zip(bars, iou_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("IoU")
    ax.set_title(f"Per-class IoU — {arch}  (Test · Model77)")
    ax.legend()
    ax.tick_params(axis="x", rotation=10)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Comparison visualisations  ← NEW
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison_bars(
    all_results: dict,
    out_path:    str | None = None,
) -> None:
    """
    Grouped bar chart comparing all architectures across four metrics:
    test_accuracy, test_dice, mean_iou, and best_val_loss.
    """
    archs   = list(all_results.keys())
    metrics = ["test_accuracy", "test_dice", "mean_iou", "best_val_loss"]
    labels  = ["Test Accuracy", "Test Dice", "Mean IoU", "Best Val Loss"]
    colors  = ["steelblue", "tomato", "seagreen", "mediumpurple"]

    x      = np.arange(len(archs))
    width  = 0.20
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [all_results[a][metric] for a in archs]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85, edgecolor="k")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([a.replace("_", "\n") for a in archs], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Architecture Comparison — Test Set (Model77)", fontsize=13)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
    plt.close(fig)


def plot_comparison_training_curves(
    all_histories: dict,
    out_path:      str | None = None,
) -> None:
    """
    Overlay training and validation loss curves for all architectures
    in a single figure so convergence speed is directly comparable.
    """
    arch_colors = {
        "unet_deep":    "steelblue",
        "segnet":       "tomato",
        "deeplab_lite": "seagreen",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for arch, history in all_histories.items():
        hist   = history.history
        epochs = range(1, len(hist["loss"]) + 1)
        color  = arch_colors.get(arch, "gray")
        axes[0].plot(epochs, hist["loss"],     lw=2, color=color,      label=f"{arch} train")
        axes[0].plot(epochs, hist["val_loss"], lw=2, color=color, ls="--", label=f"{arch} val")
        axes[1].plot(epochs, hist["dice_coef"],     lw=2, color=color,      label=f"{arch} train")
        axes[1].plot(epochs, hist["val_dice_coef"], lw=2, color=color, ls="--", label=f"{arch} val")

    axes[0].set_title("Loss (all architectures)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    axes[1].set_title("Dice Coefficient (all architectures)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    plt.suptitle("Training curves — Architecture Comparison", fontsize=13)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
    plt.close(fig)


def plot_comparison_iou_heatmap(
    all_results: dict,
    out_path:    str | None = None,
) -> None:
    """
    Heatmap of per-class IoU for every architecture — useful to spot
    which architecture handles each semantic class best.
    """
    archs      = list(all_results.keys())
    class_keys = ["class_0_iou", "class_1_iou", "class_2_iou", "mean_iou"]
    row_labels = ["Softening IoU", "Neutral IoU", "Hardening IoU", "Mean IoU"]

    matrix = np.array(
        [[all_results[a][k] for k in class_keys] for a in archs]
    )  # shape (n_archs, 4)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix.T, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="IoU")

    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels([a.replace("_", "\n") for a in archs], fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    for i in range(len(archs)):
        for j in range(len(class_keys)):
            val = matrix[i, j]
            ax.text(i, j, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color="black" if val < 0.7 else "white")

    ax.set_title("Per-class IoU Heatmap — Test Set (Model77)", fontsize=12)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
    plt.close(fig)


def plot_comparison_predictions(
    dataset:    dict,
    all_models: dict,
    sample_idx: int,
    out_dir:    str,
) -> None:
    """
    For a given test sample, show the GT mask alongside each architecture's
    prediction in a single figure — direct visual comparison.
    """
    archs     = list(all_models.keys())
    x_sample  = dataset["X_test"][sample_idx]
    y_true    = dataset["y_test_class_idx"][sample_idx]

    ncols = 2 + len(archs)   # dIP + GT + one column per arch
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.patch.set_facecolor("white")

    mask_patches = [
        mpatches.Patch(color="red",       label="0 — Softening"),
        mpatches.Patch(color="white",     label="1 — Neutral"),
        mpatches.Patch(color="royalblue", label="2 — Hardening"),
    ]

    # Panel 0 — Input dIP
    raw  = x_sample.squeeze()
    norm = plt.Normalize(vmin=-np.std(raw) * 3, vmax=np.std(raw) * 3)
    im0  = axes[0].imshow(
        raw, aspect="auto", cmap=SEISMIC_RWB, norm=norm,
        extent=[0, raw.shape[1], raw.shape[0], 0], interpolation="nearest",
    )
    plt.colorbar(im0, ax=axes[0], label="dIP")
    axes[0].set_title("Input dIP", fontsize=10)
    axes[0].set_xlabel("Crossline (W)", fontsize=8)
    axes[0].set_ylabel("Time (H)", fontsize=8)

    # Panel 1 — Ground-truth
    axes[1].imshow(
        y_true, aspect="auto", cmap=SEG_CMAP, norm=SEG_NORM,
        extent=[0, y_true.shape[1], y_true.shape[0], 0], interpolation="nearest",
    )
    axes[1].legend(handles=mask_patches, loc="lower right", fontsize=7)
    axes[1].set_title("Ground Truth", fontsize=10)
    axes[1].set_xlabel("Crossline (W)", fontsize=8)

    # Panels 2+ — one per architecture
    for col, (arch, model) in enumerate(all_models.items(), start=2):
        y_pred_probs = model.predict(
            dataset["X_test"][[sample_idx]], verbose=0
        )
        y_pred = np.argmax(y_pred_probs, axis=-1)[0]
        pct_ok = (y_true == y_pred).mean() * 100

        axes[col].imshow(
            y_pred, aspect="auto", cmap=SEG_CMAP, norm=SEG_NORM,
            extent=[0, y_pred.shape[1], y_pred.shape[0], 0], interpolation="nearest",
        )
        axes[col].legend(handles=mask_patches, loc="lower right", fontsize=7)
        axes[col].set_title(f"{arch}\n({pct_ok:.1f}% correct)", fontsize=10)
        axes[col].set_xlabel("Crossline (W)", fontsize=8)

    fig.suptitle(
        f"Architecture comparison — test sample {sample_idx}", fontsize=12
    )
    plt.tight_layout()
    out_path = str(Path(out_dir) / f"comparison_pred_sample{sample_idx}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {out_path}")
    plt.close(fig)


def print_comparison_table(all_results: dict) -> None:
    """Print a formatted comparison table to stdout."""
    archs = list(all_results.keys())
    cols  = [
        ("test_accuracy", "Accuracy"),
        ("test_dice",     "Dice"),
        ("mean_iou",      "mIoU"),
        ("class_0_iou",   "IoU Soft"),
        ("class_1_iou",   "IoU Neutral"),
        ("class_2_iou",   "IoU Hard"),
        ("best_val_loss", "ValLoss"),
        ("total_params",  "Params"),
        ("training_time_s", "Time(s)"),
    ]

    header = f"{'Architecture':<18}" + "".join(f"{lbl:>13}" for _, lbl in cols)
    sep    = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for arch in archs:
        row = f"{arch:<18}"
        for key, _ in cols:
            v = all_results[arch].get(key, "—")
            row += f"{v:>13}" if isinstance(v, str) else f"{v:>13.4f}" if isinstance(v, float) else f"{v:>13,}"
        print(row)
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4D seismic semantic segmentation — train & compare architectures"
    )
    # Data paths
    parser.add_argument("--train-ip",  default=os.getenv(
        "TRAIN_IP_PATH",
        "/home/users/jeanfranco.escobedo/jeanfranco_2025/Jeanfranco/data/raw_dfs/Model01_dIP.csv"))
    parser.add_argument("--train-amp", default=os.getenv(
        "TRAIN_AMP_PATH",
        "/home/users/jeanfranco.escobedo/jeanfranco_2025/Jeanfranco/data/raw_dfs/Model01_dAMP.csv"))
    parser.add_argument("--test-ip",   default=os.getenv(
        "TEST_IP_PATH",
        "/home/users/jeanfranco.escobedo/jeanfranco_2025/Jeanfranco/data/raw_dfs/Model77_dIP.csv"))
    parser.add_argument("--test-amp",  default=os.getenv(
        "TEST_AMP_PATH",
        "/home/users/jeanfranco.escobedo/jeanfranco_2025/Jeanfranco/data/raw_dfs/Model77_dAMP.csv"))
    # Training settings
    parser.add_argument("--architecture", choices=ARCH_NAMES, default="unet_deep",
                        help="Single architecture to train (ignored when --compare is set)")
    parser.add_argument("--compare", action="store_true",
                        help="Train and compare ALL three architectures")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--sigma",      type=float, default=0.5,
                        help="σ threshold for softening/hardening class boundaries")
    parser.add_argument("--sample-idx", type=int,   default=0,
                        help="Test sample index used for prediction visualisation")
    parser.add_argument("--output-dir", default="artifacts")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Config objects ────────────────────────────────────────────────────────
    data_config = DataConfig(
        train_ip_path  = args.train_ip,
        train_amp_path = args.train_amp,
        test_ip_path   = args.test_ip,
        test_amp_path  = args.test_amp,
    )
    training_config = TrainingConfig(
        architecture       = args.architecture,
        epochs             = args.epochs,
        batch_size         = args.batch_size,
        segmentation_sigma = args.sigma,
        output_dir         = args.output_dir,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & preprocess data ────────────────────────────────────────────────
    print("Loading data …")
    train_ip_df, train_amp_df, test_ip_df, test_amp_df = load_data(data_config)

    print("Preprocessing …")
    dataset = preprocess_for_segmentation(
        train_ip_df, train_amp_df,
        test_ip_df,  test_amp_df,
        train_col = data_config.train_value_col,
        test_col  = data_config.test_value_col,
        val_split = training_config.val_split,
        sigma     = training_config.segmentation_sigma,
    )
    print(f"  X_train={dataset['X_train'].shape}  X_test={dataset['X_test'].shape}")

    # ══════════════════════════════════════════════════════════════════════════
    # Branch A — Compare all architectures
    # ══════════════════════════════════════════════════════════════════════════
    if args.compare:
        print("\n── Running full architecture comparison ──────────────────────")
        all_results, all_histories, all_models = compare_architectures(
            dataset, training_config
        )

        # Print table
        print_comparison_table(all_results)

        # Visualisations
        print("\nGenerating comparison plots …")
        plot_comparison_bars(
            all_results,
            out_path=str(out_dir / "comparison_bars.png"),
        )
        plot_comparison_training_curves(
            all_histories,
            out_path=str(out_dir / "comparison_training_curves.png"),
        )
        plot_comparison_iou_heatmap(
            all_results,
            out_path=str(out_dir / "comparison_iou_heatmap.png"),
        )

        sample_idx = min(args.sample_idx, dataset["X_test"].shape[0] - 1)
        plot_comparison_predictions(
            dataset, all_models,
            sample_idx=sample_idx,
            out_dir=str(out_dir),
        )

        # Per-arch training curves & sample predictions
        for arch, history in all_histories.items():
            plot_training_curves(
                history, arch,
                out_path=str(out_dir / f"training_curves_{arch}.png"),
            )
            plot_iou_bars(
                all_results[arch], arch,
                out_path=str(out_dir / f"iou_bars_{arch}.png"),
            )
            y_pred_probs = all_models[arch].predict(dataset["X_test"], verbose=0)
            y_pred_idx   = np.argmax(y_pred_probs, axis=-1)
            plot_segmentation_sample(
                x_sample=dataset["X_test"][sample_idx],
                y_true  =dataset["y_test_class_idx"][sample_idx],
                y_pred  =y_pred_idx[sample_idx],
                title   =f"{arch} — test sample {sample_idx}",
                out_path=str(out_dir / f"pred_{arch}_sample{sample_idx}.png"),
            )

        print(f"\nAll artefacts written to: {out_dir}/")

    # ══════════════════════════════════════════════════════════════════════════
    # Branch B — Single architecture
    # ══════════════════════════════════════════════════════════════════════════
    else:
        arch = args.architecture
        print(f"\n── Training single architecture: {arch} ──────────────────────")
        metrics, history, model = train_and_evaluate(dataset, training_config)

        print("\n=== Final metrics ===")
        for key, value in metrics.items():
            print(f"  {key:<28}: {value}")

        sample_idx = min(args.sample_idx, dataset["X_test"].shape[0] - 1)
        y_pred_probs = model.predict(dataset["X_test"], verbose=0)
        y_pred_idx   = np.argmax(y_pred_probs, axis=-1)

        plot_training_curves(
            history, arch,
            out_path=str(out_dir / f"training_curves_{arch}.png"),
        )
        plot_iou_bars(
            metrics, arch,
            out_path=str(out_dir / f"iou_bars_{arch}.png"),
        )
        plot_segmentation_sample(
            x_sample=dataset["X_test"][sample_idx],
            y_true  =dataset["y_test_class_idx"][sample_idx],
            y_pred  =y_pred_idx[sample_idx],
            title   =f"{arch} — test sample {sample_idx}",
            out_path=str(out_dir / f"pred_{arch}_sample{sample_idx}.png"),
        )

        print(f"\nAll artefacts written to: {out_dir}/")


if __name__ == "__main__":
    main()
