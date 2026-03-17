from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


tf.random.set_seed(42)
np.random.seed(42)

NUM_CLASSES = 3
TARGET_SHAPE = (340, 580)


@dataclass(frozen=True)
class DataConfig:
    train_ip_path: str
    train_amp_path: str
    test_ip_path: str
    test_amp_path: str
    train_value_col: str = "Model01"
    test_value_col: str = "Model77"


@dataclass(frozen=True)
class TrainingConfig:
    architecture: str = "unet_deep"
    batch_size: int = 4
    epochs: int = 60
    learning_rate: float = 1e-3
    val_split: float = 0.2
    segmentation_sigma: float = 0.5
    output_dir: str = "artifacts"


@dataclass(frozen=True)
class SegmentationThresholds:
    softening_threshold: float
    hardening_threshold: float


def load_data(config: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_ip = pd.read_csv(config.train_ip_path)
    train_amp = pd.read_csv(config.train_amp_path)
    test_ip = pd.read_csv(config.test_ip_path)
    test_amp = pd.read_csv(config.test_amp_path)
    return train_ip, train_amp, test_ip, test_amp


def _validate_alignment(df_ip: pd.DataFrame, df_amp: pd.DataFrame) -> None:
    key_cols = ["inline", "xline", "time"]
    if not df_ip[key_cols].equals(df_amp[key_cols]):
        raise ValueError("Input and target dataframes are not aligned on inline/xline/time.")


def reshape_to_volumes(
    df_ip: pd.DataFrame,
    df_amp: pd.DataFrame,
    value_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape tabular seismic samples into (N, T, X) per inline."""
    _validate_alignment(df_ip, df_amp)

    inlines = sorted(df_ip["inline"].unique())
    xlines = sorted(df_ip["xline"].unique())
    times = sorted(df_ip["time"].unique())

    n_inlines, n_xlines, n_times = len(inlines), len(xlines), len(times)
    x_volume = np.zeros((n_inlines, n_times, n_xlines), dtype=np.float32)
    y_volume = np.zeros((n_inlines, n_times, n_xlines), dtype=np.float32)

    merged = pd.merge(df_ip, df_amp, on=["inline", "xline", "time"], suffixes=("_ip", "_amp"))
    ip_col = f"{value_col}_ip"
    amp_col = f"{value_col}_amp"

    if ip_col not in merged.columns or amp_col not in merged.columns:
        raise KeyError(
            f"Expected columns '{ip_col}' and '{amp_col}' after merge. "
            f"Available columns: {merged.columns.tolist()}"
        )

    for i, inline_id in enumerate(inlines):
        section = merged[merged["inline"] == inline_id]
        x_volume[i] = section.pivot(index="time", columns="xline", values=ip_col).values
        y_volume[i] = section.pivot(index="time", columns="xline", values=amp_col).values

    return x_volume, y_volume


def crop_volumes(x_data: np.ndarray, y_data: np.ndarray, target_shape: Tuple[int, int] = TARGET_SHAPE) -> Tuple[np.ndarray, np.ndarray]:
    """Crop from (N, 341, 581) to (N, 340, 580) for deeper pooling/upsampling pipelines."""
    h, w = target_shape
    if x_data.shape[1] < h or x_data.shape[2] < w:
        raise ValueError(f"Input volume shape {x_data.shape[1:]} is smaller than target shape {target_shape}.")
    return x_data[:, :h, :w], y_data[:, :h, :w]


def standardize_per_volume(x_data: np.ndarray) -> np.ndarray:
    x_data = x_data.astype(np.float32)
    means = np.mean(x_data, axis=(1, 2), keepdims=True)
    stds = np.std(x_data, axis=(1, 2), keepdims=True)
    return (x_data - means) / (stds + 1e-6)


def estimate_segmentation_thresholds(y_train_continuous: np.ndarray, sigma: float) -> SegmentationThresholds:
    scale = np.std(y_train_continuous)
    return SegmentationThresholds(
        softening_threshold=-sigma * scale,
        hardening_threshold=sigma * scale,
    )


def continuous_amp_to_classes(y_continuous: np.ndarray, thresholds: SegmentationThresholds) -> np.ndarray:
    y_classes = np.zeros_like(y_continuous, dtype=np.uint8)
    y_classes[y_continuous <= thresholds.softening_threshold] = 0
    y_classes[(y_continuous > thresholds.softening_threshold) & (y_continuous < thresholds.hardening_threshold)] = 1
    y_classes[y_continuous >= thresholds.hardening_threshold] = 2
    return y_classes


def to_one_hot(y_classes: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    return tf.keras.utils.to_categorical(y_classes, num_classes=num_classes).astype(np.float32)


def preprocess_for_segmentation(
    train_ip_df: pd.DataFrame,
    train_amp_df: pd.DataFrame,
    test_ip_df: pd.DataFrame,
    test_amp_df: pd.DataFrame,
    train_col: str,
    test_col: str,
    val_split: float,
    sigma: float,
) -> Dict[str, np.ndarray]:
    x_train_full, y_train_full_continuous = reshape_to_volumes(train_ip_df, train_amp_df, train_col)
    x_test, y_test_continuous = reshape_to_volumes(test_ip_df, test_amp_df, test_col)

    x_train_full, y_train_full_continuous = crop_volumes(x_train_full, y_train_full_continuous, TARGET_SHAPE)
    x_test, y_test_continuous = crop_volumes(x_test, y_test_continuous, TARGET_SHAPE)

    x_train, x_val, y_train_continuous, y_val_continuous = train_test_split(
        x_train_full,
        y_train_full_continuous,
        test_size=val_split,
        random_state=42,
        shuffle=True,
    )

    x_train = standardize_per_volume(x_train)
    x_val = standardize_per_volume(x_val)
    x_test = standardize_per_volume(x_test)

    thresholds = estimate_segmentation_thresholds(y_train_continuous, sigma=sigma)

    y_train_classes = continuous_amp_to_classes(y_train_continuous, thresholds)
    y_val_classes = continuous_amp_to_classes(y_val_continuous, thresholds)
    y_test_classes = continuous_amp_to_classes(y_test_continuous, thresholds)

    return {
        "X_train": x_train[..., np.newaxis],
        "X_val": x_val[..., np.newaxis],
        "X_test": x_test[..., np.newaxis],
        "y_train": to_one_hot(y_train_classes),
        "y_val": to_one_hot(y_val_classes),
        "y_test": to_one_hot(y_test_classes),
        "y_train_class_idx": y_train_classes,
        "y_val_class_idx": y_val_classes,
        "y_test_class_idx": y_test_classes,
        "thresholds": thresholds,
    }


def conv_block(x: tf.Tensor, filters: int, kernel_size: int = 3, dropout: float = 0.0) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    return x


def build_unet_deep(input_shape: Tuple[int, int, int], num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    c1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPool2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPool2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = tf.keras.layers.MaxPool2D()(c3)

    c4 = conv_block(p3, 256, dropout=0.2)
    p4 = tf.keras.layers.MaxPool2D()(c4)

    b = conv_block(p4, 512, dropout=0.3)

    u4 = tf.keras.layers.UpSampling2D()(b)
    u4 = tf.keras.layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, 256)

    u3 = tf.keras.layers.UpSampling2D()(c5)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, 128)

    u2 = tf.keras.layers.UpSampling2D()(c6)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, 64)

    u1 = tf.keras.layers.UpSampling2D()(c7)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, 32)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(c8)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="seismic_unet_deep")


def build_segnet(input_shape: Tuple[int, int, int], num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    e1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPool2D()(e1)
    e2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPool2D()(e2)
    e3 = conv_block(p2, 128)
    p3 = tf.keras.layers.MaxPool2D()(e3)
    e4 = conv_block(p3, 256)
    p4 = tf.keras.layers.MaxPool2D()(e4)

    d4 = tf.keras.layers.UpSampling2D()(p4)
    d4 = conv_block(d4, 256)
    d3 = tf.keras.layers.UpSampling2D()(d4)
    d3 = conv_block(d3, 128)
    d2 = tf.keras.layers.UpSampling2D()(d3)
    d2 = conv_block(d2, 64)
    d1 = tf.keras.layers.UpSampling2D()(d2)
    d1 = conv_block(d1, 32)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(d1)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="seismic_segnet")


def aspp_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    b0 = tf.keras.layers.Conv2D(filters, 1, padding="same", activation="relu")(x)
    b1 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=2, padding="same", activation="relu")(x)
    b2 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=4, padding="same", activation="relu")(x)
    b3 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=6, padding="same", activation="relu")(x)
    x = tf.keras.layers.Concatenate()([b0, b1, b2, b3])
    return tf.keras.layers.Conv2D(filters, 1, padding="same", activation="relu")(x)


def build_deeplab_lite(input_shape: Tuple[int, int, int], num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    x = conv_block(inputs, 32)
    low_level = x
    x = tf.keras.layers.MaxPool2D()(x)
    x = conv_block(x, 64)
    x = tf.keras.layers.MaxPool2D()(x)
    x = conv_block(x, 128)
    x = tf.keras.layers.MaxPool2D()(x)

    x = aspp_block(x, 128)
    x = tf.keras.layers.UpSampling2D(size=(4, 4))(x)

    low = tf.keras.layers.Conv2D(48, 1, padding="same", activation="relu")(low_level)
    x = tf.keras.layers.Concatenate()([x, low])
    x = conv_block(x, 64)
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="seismic_deeplab_lite")


def build_model(architecture: str, input_shape: Tuple[int, int, int], num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    architecture = architecture.lower()
    builders = {
        "unet_deep": build_unet_deep,
        "segnet": build_segnet,
        "deeplab_lite": build_deeplab_lite,
    }
    if architecture not in builders:
        raise ValueError(f"Unsupported architecture '{architecture}'. Use one of: {list(builders.keys())}.")
    return builders[architecture](input_shape, num_classes)


def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denominator = tf.reduce_sum(y_true_f + y_pred_f, axis=0)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(dice)


def compute_iou_per_class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> Dict[str, float]:
    ious: Dict[str, float] = {}
    for cls in range(num_classes):
        true_mask = y_true == cls
        pred_mask = y_pred == cls
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        ious[f"class_{cls}_iou"] = float(intersection / (union + 1e-6))
    ious["mean_iou"] = float(np.mean([ious[f"class_{cls}_iou"] for cls in range(num_classes)]))
    return ious


def train_and_evaluate(data: Dict[str, np.ndarray], config: TrainingConfig) -> Dict[str, float]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config.architecture, input_shape=data["X_train"].shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", dice_coef],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f"best_{config.architecture}.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        batch_size=config.batch_size,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc, test_dice = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    y_pred_probs = model.predict(data["X_test"], verbose=0)

    y_true = np.argmax(data["y_test"], axis=-1)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    metrics = {
        "input_shape": str(data["X_train"].shape[1:]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_dice": float(test_dice),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "best_val_accuracy": float(np.max(history.history["val_accuracy"])),
    }
    metrics.update(compute_iou_per_class(y_true, y_pred, num_classes=NUM_CLASSES))

    with open(output_dir / "metrics.txt", "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    return metrics
