from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from src.model import DataConfig, TrainingConfig, preprocess_for_segmentation, train_and_evaluate


def plot_segmentation_pair(
    x_sample: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray | None = None,
    title: str = "Segmentation sample",
    out_path: str | None = None,
) -> None:
    cmap = ListedColormap(["red", "lightgray", "blue"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    ncols = 3 if y_pred is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    axes[0].imshow(x_sample.squeeze(), aspect="auto", cmap="seismic")
    axes[0].set_title("Input dIP")
    axes[0].set_xlabel("Crossline")
    axes[0].set_ylabel("Time")

    axes[1].imshow(y_true, aspect="auto", cmap=cmap, norm=norm)
    axes[1].set_title("True class map\n(0=soft, 1=neutral, 2=hard)")
    axes[1].set_xlabel("Crossline")

    if y_pred is not None:
        axes[2].imshow(y_pred, aspect="auto", cmap=cmap, norm=norm)
        axes[2].set_title("Predicted class map")
        axes[2].set_xlabel("Crossline")

    fig.suptitle(title)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4D seismic hardening/softening segmentation")
    parser.add_argument("--train-ip", default=os.getenv("TRAIN_IP_PATH", "/home/users/jeanfranco.escobedo/jeanfranco_2025/Jeanfranco/data/raw_dfs/Model01_dIP.csv"))
    parser.add_argument("--train-amp", default=os.getenv("TRAIN_AMP_PATH", "/home/users/jeanfranco.escobedo/jeanfranco_2025/Jeanfranco/data/raw_dfs/Model01_dAMP.csv"))
    parser.add_argument("--test-ip", default=os.getenv("TEST_IP_PATH", "/home/users/jeanfranco.escobedo/jeanfranco_2025/Jeanfranco/data/raw_dfs/Model77_dIP.csv"))
    parser.add_argument("--test-amp", default=os.getenv("TEST_AMP_PATH", "/home/users/jeanfranco.escobedo/jeanfranco_2025/Jeanfranco/data/raw_dfs/Model77_dAMP.csv"))
    parser.add_argument("--architecture", choices=["unet", "fcn"], default="unet")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.5, help="Threshold scale for converting dAmp to classes")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--output-dir", default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_config = DataConfig(
        train_ip_path=args.train_ip,
        train_amp_path=args.train_amp,
        test_ip_path=args.test_ip,
        test_amp_path=args.test_amp,
    )
    training_config = TrainingConfig(
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        segmentation_sigma=args.sigma,
        output_dir=args.output_dir,
    )

    from src.model import load_data

    train_ip_df, train_amp_df, test_ip_df, test_amp_df = load_data(data_config)
    dataset = preprocess_for_segmentation(
        train_ip_df,
        train_amp_df,
        test_ip_df,
        test_amp_df,
        train_col=data_config.train_value_col,
        test_col=data_config.test_value_col,
        val_split=training_config.val_split,
        sigma=training_config.segmentation_sigma,
    )

    metrics = train_and_evaluate(dataset, training_config)
    print("\n=== Final metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    idx = min(args.sample_idx, dataset["X_test"].shape[0] - 1)
    plot_segmentation_pair(
        x_sample=dataset["X_test"][idx],
        y_true=dataset["y_test_class_idx"][idx],
        title=f"Model77 sample {idx} ground truth",
        out_path=f"{args.output_dir}/sample_{idx}_truth.png",
    )


if __name__ == "__main__":
    main()
