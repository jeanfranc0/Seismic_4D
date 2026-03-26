"""
src/model.py
============
4D Seismic Hardening / Softening — Semantic Segmentation
---------------------------------------------------------
Responsibilities
  - Data loading & preprocessing (reshape, crop, standardise, class labels)
  - Three segmentation architectures: UNet-Deep, SegNet, DeepLab-lite
  - Training helpers: combined loss (CE + Dice), IoU metric
  - Single-architecture  train_and_evaluate()
  - Multi-architecture   compare_architectures()  ← NEW
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ── Reproducibility ───────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── Global constants ──────────────────────────────────────────────────────────
NUM_CLASSES  = 3
TARGET_SHAPE = (340, 580)   # cropped from raw (341, 581)
ARCH_NAMES: List[str] = ["unet_deep", "segnet", "deeplab_lite"]


# ══════════════════════════════════════════════════════════════════════════════
# Config dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DataConfig:
    train_ip_path:   str
    train_amp_path:  str
    test_ip_path:    str
    test_amp_path:   str
    train_value_col: str = "Model01"
    test_value_col:  str = "Model77"


@dataclass(frozen=True)
class TrainingConfig:
    architecture:       str   = "unet_deep"   # one of ARCH_NAMES
    batch_size:         int   = 4
    epochs:             int   = 60
    learning_rate:      float = 1e-3
    val_split:          float = 0.2
    segmentation_sigma: float = 0.5           # σ threshold for class boundaries
    output_dir:         str   = "artifacts"


@dataclass(frozen=True)
class SegmentationThresholds:
    softening_threshold: float
    hardening_threshold: float


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data(
    config: DataConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read all four CSV files and return them as DataFrames."""
    train_ip  = pd.read_csv(config.train_ip_path)
    train_amp = pd.read_csv(config.train_amp_path)
    test_ip   = pd.read_csv(config.test_ip_path)
    test_amp  = pd.read_csv(config.test_amp_path)
    return train_ip, train_amp, test_ip, test_amp


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _validate_alignment(df_ip: pd.DataFrame, df_amp: pd.DataFrame) -> None:
    key_cols = ["inline", "xline", "time"]
    if not df_ip[key_cols].equals(df_amp[key_cols]):
        raise ValueError("IP and AMP dataframes are not aligned on inline/xline/time.")


def _resolve_signal_column(df: pd.DataFrame, model_tag: str, signal_hint: str) -> str:
    """
    Flexibly resolve which column holds the seismic value.

    Supported naming conventions
    ----------------------------
    - Model01_dIP / Model01_dAMP
    - Model01_ip  / Model01_amp
    - single value column (fallback)
    """
    key_cols   = {"inline", "xline", "time"}
    value_cols = [c for c in df.columns if c not in key_cols]
    hint       = signal_hint.lower()
    tag_l      = model_tag.lower()

    if model_tag in df.columns:
        return model_tag
    for pat in [f"{model_tag}_d{hint}", f"{model_tag}_{hint}"]:
        if pat in df.columns:
            return pat
    filtered = [c for c in value_cols if tag_l in c.lower() and hint in c.lower()]
    if len(filtered) == 1:
        return filtered[0]
    if len(value_cols) == 1:
        return value_cols[0]
    raise KeyError(
        f"Cannot resolve column for model_tag='{model_tag}', hint='{signal_hint}'. "
        f"Available value columns: {value_cols}"
    )


def reshape_to_volumes(
    df_ip:     pd.DataFrame,
    df_amp:    pd.DataFrame,
    value_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pivot flat tabular seismic data into (N_inlines, T, X) volume arrays."""
    _validate_alignment(df_ip, df_amp)

    inlines = sorted(df_ip["inline"].unique())
    times   = sorted(df_ip["time"].unique())
    xlines  = sorted(df_ip["xline"].unique())

    n_i, n_t, n_x = len(inlines), len(times), len(xlines)
    x_vol = np.zeros((n_i, n_t, n_x), dtype=np.float32)
    y_vol = np.zeros((n_i, n_t, n_x), dtype=np.float32)

    ip_col  = _resolve_signal_column(df_ip,  value_col, "ip")
    amp_col = _resolve_signal_column(df_amp, value_col, "amp")

    for i, inline_id in enumerate(inlines):
        sec_ip  = df_ip [df_ip ["inline"] == inline_id]
        sec_amp = df_amp[df_amp["inline"] == inline_id]
        x_vol[i] = sec_ip .pivot(index="time", columns="xline", values=ip_col ).values
        y_vol[i] = sec_amp.pivot(index="time", columns="xline", values=amp_col).values

    return x_vol, y_vol


def crop_volumes(
    x_data: np.ndarray,
    y_data: np.ndarray,
    target_shape: Tuple[int, int] = TARGET_SHAPE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop (N,341,581) → (N,340,580) for clean power-of-2 pooling stacks."""
    h, w = target_shape
    if x_data.shape[1] < h or x_data.shape[2] < w:
        raise ValueError(
            f"Volume shape {x_data.shape[1:]} is smaller than target {target_shape}."
        )
    return x_data[:, :h, :w], y_data[:, :h, :w]


def standardize_per_volume(x_data: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalisation computed independently per inline."""
    x_data = x_data.astype(np.float32)
    means  = np.mean(x_data, axis=(1, 2), keepdims=True)
    stds   = np.std( x_data, axis=(1, 2), keepdims=True)
    return (x_data - means) / (stds + 1e-6)


def estimate_segmentation_thresholds(
    y_train_continuous: np.ndarray,
    sigma: float,
) -> SegmentationThresholds:
    """Compute ±σ·std(y_train) thresholds for the three semantic classes."""
    scale = np.std(y_train_continuous)
    return SegmentationThresholds(
        softening_threshold=-sigma * scale,
        hardening_threshold= sigma * scale,
    )


def continuous_amp_to_classes(
    y_continuous: np.ndarray,
    thresholds:   SegmentationThresholds,
) -> np.ndarray:
    """
    Map continuous dAMP values to class indices.
    0 = softening  (y <= lower threshold)
    1 = neutral    (lower < y < upper)
    2 = hardening  (y >= upper threshold)
    """
    y_classes = np.ones_like(y_continuous, dtype=np.uint8)   # default: neutral
    y_classes[y_continuous <= thresholds.softening_threshold] = 0
    y_classes[y_continuous >= thresholds.hardening_threshold] = 2
    return y_classes


def to_one_hot(y_classes: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    return tf.keras.utils.to_categorical(y_classes, num_classes=num_classes).astype(np.float32)


def preprocess_for_segmentation(
    train_ip_df:  pd.DataFrame,
    train_amp_df: pd.DataFrame,
    test_ip_df:   pd.DataFrame,
    test_amp_df:  pd.DataFrame,
    train_col:    str,
    test_col:     str,
    val_split:    float,
    sigma:        float,
) -> Dict[str, np.ndarray]:
    """
    Full preprocessing pipeline.

    Returns a dict with keys:
      X_train, X_val, X_test          — (N, H, W, 1)  float32
      y_train, y_val, y_test          — (N, H, W, 3)  one-hot float32
      y_train_class_idx, ...          — (N, H, W)     uint8
      thresholds                      — SegmentationThresholds
    """
    # 1. Reshape CSV tables → 3-D volumes
    x_train_full, y_train_cont = reshape_to_volumes(train_ip_df, train_amp_df, train_col)
    x_test,       y_test_cont  = reshape_to_volumes(test_ip_df,  test_amp_df,  test_col)

    # 2. Crop to (340, 580)
    x_train_full, y_train_cont = crop_volumes(x_train_full, y_train_cont, TARGET_SHAPE)
    x_test,       y_test_cont  = crop_volumes(x_test,       y_test_cont,  TARGET_SHAPE)

    # 3. Train / validation split
    x_train, x_val, y_tr_cont, y_val_cont = train_test_split(
        x_train_full, y_train_cont,
        test_size=val_split, random_state=42, shuffle=True,
    )

    # 4. Per-volume standardisation
    x_train = standardize_per_volume(x_train)
    x_val   = standardize_per_volume(x_val)
    x_test  = standardize_per_volume(x_test)

    # 5. Estimate thresholds from training set only (no data leakage)
    thresholds = estimate_segmentation_thresholds(y_tr_cont, sigma=sigma)

    # 6. Convert continuous amplitudes → class indices
    y_train_cls = continuous_amp_to_classes(y_tr_cont,  thresholds)
    y_val_cls   = continuous_amp_to_classes(y_val_cont, thresholds)
    y_test_cls  = continuous_amp_to_classes(y_test_cont, thresholds)

    return {
        "X_train":          x_train[..., np.newaxis],
        "X_val":            x_val  [..., np.newaxis],
        "X_test":           x_test [..., np.newaxis],
        "y_train":          to_one_hot(y_train_cls),
        "y_val":            to_one_hot(y_val_cls),
        "y_test":           to_one_hot(y_test_cls),
        "y_train_class_idx": y_train_cls,
        "y_val_class_idx":   y_val_cls,
        "y_test_class_idx":  y_test_cls,
        "thresholds":        thresholds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Shared building blocks
# ══════════════════════════════════════════════════════════════════════════════

def conv_block(
    x:           tf.Tensor,
    filters:     int,
    kernel_size: int   = 3,
    dropout:     float = 0.0,
) -> tf.Tensor:
    """
    Double Conv2D → BN → ReLU block.
    Uses SpatialDropout2D (drops entire feature maps) instead of regular
    Dropout — better suited for spatial segmentation tasks.
    """
    for _ in range(2):
        x = tf.keras.layers.Conv2D(
            filters, kernel_size, padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
    if dropout > 0:
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
    return x


def resize_like(source: tf.Tensor, reference: tf.Tensor, name: str) -> tf.Tensor:
    """Bilinearly resize `source` to the spatial dimensions of `reference`."""
    return tf.keras.layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3], method="bilinear"),
        name=name,
    )([source, reference])


def resize_to_shape(
    source:       tf.Tensor,
    target_shape: Tuple[int, int],
    name:         str,
) -> tf.Tensor:
    return tf.keras.layers.Resizing(
        target_shape[0], target_shape[1], interpolation="bilinear", name=name
    )(source)


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 1 — U-Net Deep
# ══════════════════════════════════════════════════════════════════════════════

def build_unet_deep(
    input_shape: Tuple[int, int, int],
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    """
    4-level U-Net with skip connections.
    Encoder filters: 32 → 64 → 128 → 256, bottleneck: 512.
    Skip connections give the decoder both high-level semantics AND
    fine spatial detail, making this ideal for precise boundary detection.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # ── Encoder ──────────────────────────────────────────────────────────────
    c1 = conv_block(inputs, 32);              p1 = tf.keras.layers.MaxPool2D()(c1)
    c2 = conv_block(p1,     64);              p2 = tf.keras.layers.MaxPool2D()(c2)
    c3 = conv_block(p2,    128);              p3 = tf.keras.layers.MaxPool2D()(c3)
    c4 = conv_block(p3,    256, dropout=0.2); p4 = tf.keras.layers.MaxPool2D()(c4)

    # ── Bottleneck ────────────────────────────────────────────────────────────
    b = conv_block(p4, 512, dropout=0.3)

    # ── Decoder with skip connections ─────────────────────────────────────────
    u4 = tf.keras.layers.UpSampling2D()(b)
    u4 = resize_like(u4, c4, name="unet_resize_u4")
    u4 = tf.keras.layers.Concatenate()([u4, c4])
    d4 = conv_block(u4, 256)

    u3 = tf.keras.layers.UpSampling2D()(d4)
    u3 = resize_like(u3, c3, name="unet_resize_u3")
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    d3 = conv_block(u3, 128)

    u2 = tf.keras.layers.UpSampling2D()(d3)
    u2 = resize_like(u2, c2, name="unet_resize_u2")
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, 64)

    u1 = tf.keras.layers.UpSampling2D()(d2)
    u1 = resize_like(u1, c1, name="unet_resize_u1")
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    d1 = conv_block(u1, 32)

    # ── Output — cast to float32 for mixed-precision safety ───────────────────
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax", dtype="float32")(d1)
    outputs = resize_to_shape(outputs, input_shape[:2], name="unet_output_resize")
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="seismic_unet_deep")


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 2 — SegNet
# ══════════════════════════════════════════════════════════════════════════════

def build_segnet(
    input_shape: Tuple[int, int, int],
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    """
    Encoder–decoder WITHOUT skip connections (pure SegNet style).
    The decoder must reconstruct spatial detail purely from the bottleneck.
    Fewer parameters than U-Net; faster but potentially less precise boundaries.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # ── Encoder ──────────────────────────────────────────────────────────────
    e1 = conv_block(inputs, 32);  p1 = tf.keras.layers.MaxPool2D()(e1)
    e2 = conv_block(p1,     64);  p2 = tf.keras.layers.MaxPool2D()(e2)
    e3 = conv_block(p2,    128);  p3 = tf.keras.layers.MaxPool2D()(e3)
    e4 = conv_block(p3,    256);  p4 = tf.keras.layers.MaxPool2D()(e4)

    # ── Decoder (no skip connections) ─────────────────────────────────────────
    d4 = tf.keras.layers.UpSampling2D()(p4)
    d4 = resize_like(d4, e4, name="segnet_resize_d4")
    d4 = conv_block(d4, 256)

    d3 = tf.keras.layers.UpSampling2D()(d4)
    d3 = resize_like(d3, e3, name="segnet_resize_d3")
    d3 = conv_block(d3, 128)

    d2 = tf.keras.layers.UpSampling2D()(d3)
    d2 = resize_like(d2, e2, name="segnet_resize_d2")
    d2 = conv_block(d2, 64)

    d1 = tf.keras.layers.UpSampling2D()(d2)
    d1 = resize_like(d1, e1, name="segnet_resize_d1")
    d1 = conv_block(d1, 32)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax", dtype="float32")(d1)
    outputs = resize_to_shape(outputs, input_shape[:2], name="segnet_output_resize")
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="seismic_segnet")


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 3 — DeepLab-lite (ASPP)
# ══════════════════════════════════════════════════════════════════════════════

def aspp_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """
    Atrous Spatial Pyramid Pooling.
    Applies 4 parallel convolutions with different dilation rates so the
    network simultaneously captures fine detail (rate=1) and broad context
    (rates=2,4,6) without losing resolution.
    """
    b0 = tf.keras.layers.Conv2D(filters, 1,                    padding="same", activation="relu")(x)
    b1 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=2,   padding="same", activation="relu")(x)
    b2 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=4,   padding="same", activation="relu")(x)
    b3 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=6,   padding="same", activation="relu")(x)
    fused = tf.keras.layers.Concatenate()([b0, b1, b2, b3])
    return tf.keras.layers.Conv2D(filters, 1, padding="same", activation="relu")(fused)


def build_deeplab_lite(
    input_shape: Tuple[int, int, int],
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    """
    Lightweight DeepLab v3 variant.
    3-level backbone + ASPP module + low-level feature fusion.
    Multi-scale context from ASPP makes this robust to features of
    varying spatial extent — important for seismic signal detection.
    """
    inputs    = tf.keras.Input(shape=input_shape)
    low_level = conv_block(inputs, 32)   # full-resolution feature map saved for fusion

    x = tf.keras.layers.MaxPool2D()(low_level)
    x = conv_block(x, 64)
    x = tf.keras.layers.MaxPool2D()(x)
    x = conv_block(x, 128)
    x = tf.keras.layers.MaxPool2D()(x)

    # ASPP + upsample back to low-level resolution
    x = aspp_block(x, 128)
    x = tf.keras.layers.UpSampling2D(size=(4, 4))(x)
    x = resize_like(x, low_level, name="deeplab_resize_decoder")

    # Fuse with low-level features
    low = tf.keras.layers.Conv2D(48, 1, padding="same", activation="relu")(low_level)
    x   = tf.keras.layers.Concatenate()([x, low])
    x   = conv_block(x, 64)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax", dtype="float32")(x)
    outputs = resize_to_shape(outputs, input_shape[:2], name="deeplab_output_resize")
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="seismic_deeplab_lite")


# ══════════════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════════════

_BUILDERS = {
    "unet_deep":    build_unet_deep,
    "segnet":       build_segnet,
    "deeplab_lite": build_deeplab_lite,
}


def build_model(
    architecture: str,
    input_shape:  Tuple[int, int, int],
    num_classes:  int = NUM_CLASSES,
) -> tf.keras.Model:
    architecture = architecture.lower()
    if architecture not in _BUILDERS:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: {list(_BUILDERS.keys())}"
        )
    return _BUILDERS[architecture](input_shape, num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions & metrics
# ══════════════════════════════════════════════════════════════════════════════

def dice_coef(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """Soft Dice coefficient averaged across all classes."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    yt = tf.reshape(y_true, [-1, NUM_CLASSES])
    yp = tf.reshape(y_pred, [-1, NUM_CLASSES])
    num = 2.0 * tf.reduce_sum(yt * yp, axis=0) + smooth
    den = tf.reduce_sum(yt + yp, axis=0) + smooth
    return tf.reduce_mean(num / den)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 1.0 - dice_coef(y_true, y_pred)


def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    50% categorical cross-entropy + 50% Dice loss.
    CE handles per-pixel log-likelihood; Dice handles class imbalance.
    """
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(ce) + dice_loss(y_true, y_pred)


def compute_iou_per_class(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """Compute IoU for each class and the mean IoU."""
    ious: Dict[str, float] = {}
    for cls in range(num_classes):
        tp = np.logical_and(y_true == cls, y_pred == cls).sum()
        fp = np.logical_and(y_true != cls, y_pred == cls).sum()
        fn = np.logical_and(y_true == cls, y_pred != cls).sum()
        ious[f"class_{cls}_iou"] = float(tp / (tp + fp + fn + 1e-6))
    ious["mean_iou"] = float(np.mean(list(ious.values())))
    return ious


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_callbacks(config: TrainingConfig, arch: str) -> list:
    output_dir = Path(config.output_dir)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f"best_{arch}.keras"),
            monitor="val_loss", save_best_only=True, verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(
            str(output_dir / f"training_log_{arch}.csv")
        ),
    ]


def train_and_evaluate(
    data:   Dict[str, np.ndarray],
    config: TrainingConfig,
) -> Dict[str, float]:
    """
    Train a single architecture defined by config.architecture,
    evaluate on the test set, and return a metrics dict.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arch  = config.architecture
    model = build_model(arch, input_shape=data["X_train"].shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=combined_loss,
        metrics=["accuracy", dice_coef],
    )

    t0 = time.time()
    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        batch_size=config.batch_size,
        epochs=config.epochs,
        callbacks=_make_callbacks(config, arch),
        verbose=1,
    )
    training_time = time.time() - t0

    test_loss, test_acc, test_dice = model.evaluate(
        data["X_test"], data["y_test"], verbose=0
    )
    y_pred_probs = model.predict(data["X_test"], verbose=0)
    y_true = np.argmax(data["y_test"], axis=-1)
    y_pred = np.argmax(y_pred_probs,   axis=-1)

    metrics: Dict[str, float] = {
        "architecture":      arch,
        "input_shape":       str(data["X_train"].shape[1:]),
        "total_params":      int(model.count_params()),
        "training_time_s":   round(training_time, 1),
        "epochs_trained":    len(history.history["loss"]),
        "test_loss":         round(float(test_loss), 4),
        "test_accuracy":     round(float(test_acc),  4),
        "test_dice":         round(float(test_dice), 4),
        "best_val_loss":     round(float(np.min(history.history["val_loss"])), 4),
        "best_val_accuracy": round(float(np.max(history.history["val_accuracy"])), 4),
    }
    metrics.update(compute_iou_per_class(y_true, y_pred))

    # Persist per-architecture metrics
    metrics_path = output_dir / f"metrics_{arch}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics, history, model


# ══════════════════════════════════════════════════════════════════════════════
# Multi-architecture comparison  ← NEW
# ══════════════════════════════════════════════════════════════════════════════

def compare_architectures(
    data:          Dict[str, np.ndarray],
    config:        TrainingConfig,
    architectures: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate every requested architecture on the same dataset,
    then return a nested dict  {arch_name: metrics_dict}.

    Parameters
    ----------
    data          : preprocessed dataset dict from preprocess_for_segmentation
    config        : TrainingConfig — architecture field is IGNORED (each arch
                    is trained in turn); all other settings are reused
    architectures : list of arch names to run; defaults to all three

    Returns
    -------
    all_results : dict  {arch_name → metrics_dict}
                  Also writes  artifacts/comparison_results.json
    """
    if architectures is None:
        architectures = ARCH_NAMES

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results:  Dict[str, Dict]              = {}
    all_histories: Dict[str, tf.keras.callbacks.History] = {}
    all_models:   Dict[str, tf.keras.Model]   = {}

    for arch in architectures:
        print(f"\n{'=' * 60}")
        print(f"  Training architecture: {arch.upper()}")
        print(f"{'=' * 60}")

        # Swap architecture in a new config (all other settings unchanged)
        arch_config = TrainingConfig(
            architecture       = arch,
            batch_size         = config.batch_size,
            epochs             = config.epochs,
            learning_rate      = config.learning_rate,
            val_split          = config.val_split,
            segmentation_sigma = config.segmentation_sigma,
            output_dir         = config.output_dir,
        )

        metrics, history, model = train_and_evaluate(data, arch_config)
        all_results[arch]   = metrics
        all_histories[arch] = history
        all_models[arch]    = model

        print(f"  → test_acc={metrics['test_accuracy']:.4f}  "
              f"dice={metrics['test_dice']:.4f}  "
              f"mIoU={metrics['mean_iou']:.4f}  "
              f"time={metrics['training_time_s']:.0f}s")

    # Persist combined comparison table
    comparison_path = output_dir / "comparison_results.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComparison results saved → {comparison_path}")

    return all_results, all_histories, all_models
