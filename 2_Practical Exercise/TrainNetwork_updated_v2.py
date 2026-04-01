#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Live CNN trainer with two teaching screens.

Workshop flow:
1. Press C a few times and draw boxes around the target object.
2. Press S to start the slow first epoch.
3. Watch the live camera screen while the second screen shows the CNN.
4. Every 10 epochs the network pauses for feedback.
   - T = yes, this crop is the target object
   - P = no, this crop is background / a wrong guess
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import random
from pathlib import Path
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workshop_setup import check_required_packages

check_required_packages(
    [
        ("cv2", "opencv-python", "camera input and the two workshop windows"),
        ("numpy", "numpy", "array handling for the live visualizations"),
        ("torch", "torch", "training the small CNN live in the workshop"),
    ],
    "TrainNetwork_updated_v2.py",
)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
torch.set_grad_enabled(True)

print("[setup] This optional extension trains its own small model from scratch, so it does not download the YOLO workshop weights.")


LIVE_WINDOW = "Live Training Screen"
CNN_WINDOW = "Inside the CNN"
ANNOTATE_WINDOW = "Draw a box around the object"

TITLE_BLUE = (65, 48, 11)
ACCENT_ORANGE = (51, 121, 196)
SOFT_BG = (246, 248, 251)
PANEL_BG = (232, 236, 241)
GRID_BG = (220, 224, 228)
TEXT_DARK = (46, 46, 46)
TEXT_LIGHT = (245, 245, 245)
GOOD_GREEN = (60, 180, 90)
WARN_YELLOW = (30, 185, 240)
BAD_RED = (70, 80, 220)
LINK_BLUE = (200, 130, 40)
PURPLE = (190, 120, 120)
BACKPROP_RED = (40, 40, 220)
NEUTRAL_BOX = (180, 140, 80)


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Monitor:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Sample:
    crop: np.ndarray
    label: int
    source: str
    weight: float = 1.0
    bbox_size: int = 64


@dataclass
class ScanResult:
    boxes: List[Tuple[int, int, int, int]]
    scores: List[float]
    best_box: Optional[Tuple[int, int, int, int]]
    display_box: Optional[Tuple[int, int, int, int]]
    best_score: float
    best_crop: Optional[np.ndarray]
    checked_at: float = field(default_factory=time.monotonic)


def safe_ascii(text: str) -> str:
    return str(text).replace("–", "-").replace("—", "-").replace("…", "...")


def put_text(
    image: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    scale: float = 0.7,
    color: Tuple[int, int, int] = TEXT_DARK,
    thickness: int = 2,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    cv2.putText(
        image,
        safe_ascii(text),
        xy,
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(
    image: np.ndarray,
    rect: Rect,
    title: str = "",
    color: Tuple[int, int, int] = PANEL_BG,
    border: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    x, y, w, h = rect.x, rect.y, rect.w, rect.h
    cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), border, 2)
    if title:
        put_text(image, title, (x + 12, y + 28), scale=0.78, color=TITLE_BLUE, thickness=2)


def wrap_lines(text: str, width: int) -> List[str]:
    return textwrap.wrap(safe_ascii(text), width=width) or [safe_ascii(text)]


def draw_multiline(
    image: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    line_height: int = 24,
    width: int = 40,
    scale: float = 0.58,
    color: Tuple[int, int, int] = TEXT_DARK,
    thickness: int = 1,
) -> int:
    x, y = xy
    lines = wrap_lines(text, width)
    for i, line in enumerate(lines):
        put_text(image, line, (x, y + i * line_height), scale=scale, color=color, thickness=thickness)
    return y + len(lines) * line_height


def normalize_to_u8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((8, 8), dtype=np.uint8)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.full(arr.shape, 127, dtype=np.uint8)
    arr = (arr - lo) / (hi - lo)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def make_square_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int], pad: float = 0.18) -> Tuple[np.ndarray, int]:
    x, y, w, h = bbox
    side = max(24, int(max(w, h) * (1.0 + pad)))
    cx = x + w * 0.5
    cy = y + h * 0.5
    x0 = int(round(cx - side * 0.5))
    y0 = int(round(cy - side * 0.5))
    x1 = x0 + side
    y1 = y0 + side

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(frame.shape[1], x1)
    src_y1 = min(frame.shape[0], y1)

    crop = np.full((side, side, 3), 235, dtype=np.uint8)
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    crop[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = frame[src_y0:src_y1, src_x0:src_x1]
    return crop, side


def resize_square(image: np.ndarray, size: int = 64) -> np.ndarray:
    if image.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def crop_to_tensor(image: np.ndarray, size: int = 64) -> torch.Tensor:
    square = resize_square(image, size=size)
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
    return tensor


def iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / float(max(1, union))


def random_negative_crops(
    frame: np.ndarray,
    positive_bbox: Tuple[int, int, int, int],
    side: int,
    count: int = 6,
) -> List[np.ndarray]:
    crops: List[np.ndarray] = []
    h, w = frame.shape[:2]
    side = max(24, min(side, min(w, h)))
    margin = int(side * 0.12)
    expanded = (
        max(0, positive_bbox[0] - margin),
        max(0, positive_bbox[1] - margin),
        min(w - positive_bbox[0] + margin, positive_bbox[2] + 2 * margin),
        min(h - positive_bbox[1] + margin, positive_bbox[3] + 2 * margin),
    )
    attempts = 0
    while len(crops) < count and attempts < count * 18:
        attempts += 1
        x0 = random.randint(0, max(0, w - side))
        y0 = random.randint(0, max(0, h - side))
        candidate = (x0, y0, side, side)
        if iou_xywh(candidate, expanded) > 0.05:
            continue
        patch = frame[y0 : y0 + side, x0 : x0 + side].copy()
        if patch.size == 0:
            continue
        crops.append(patch)
    return crops


def jitter_crop(image: np.ndarray, output_size: int = 64, heavy: bool = True) -> np.ndarray:
    base = resize_square(image, output_size)
    out = base.copy()
    h, w = out.shape[:2]

    angle = random.uniform(-42.0, 42.0) if heavy else random.uniform(-22.0, 22.0)
    scale = random.uniform(0.76, 1.26) if heavy else random.uniform(0.88, 1.12)
    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    shift = 8.0 if heavy else 4.0
    matrix[0, 2] += random.uniform(-shift, shift)
    matrix[1, 2] += random.uniform(-shift, shift)
    out = cv2.warpAffine(out, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    if random.random() < (0.72 if heavy else 0.32):
        shear = random.uniform(-0.22, 0.22) if heavy else random.uniform(-0.10, 0.10)
        shear_m = np.float32([[1.0, shear, -0.5 * shear * w], [0.0, 1.0, 0.0]])
        out = cv2.warpAffine(out, shear_m, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    if random.random() < 0.9:
        alpha = random.uniform(0.58, 1.45) if heavy else random.uniform(0.75, 1.25)
        beta = random.uniform(-42.0, 42.0) if heavy else random.uniform(-20.0, 20.0)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if random.random() < 0.85:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-10, 10)) % 180
        hsv[..., 1] *= random.uniform(0.50, 1.50) if heavy else random.uniform(0.75, 1.25)
        hsv[..., 2] *= random.uniform(0.60, 1.35) if heavy else random.uniform(0.80, 1.20)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if random.random() < (0.55 if heavy else 0.30):
        k = random.choice([3, 5, 7] if heavy else [3, 5])
        out = cv2.GaussianBlur(out, (k, k), sigmaX=0)

    if random.random() < (0.28 if heavy else 0.14):
        noise = np.random.normal(0.0, 16.0 if heavy else 8.0, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < (0.34 if heavy else 0.18):
        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        dst = src + np.float32(
            [
                [random.uniform(-4, 4), random.uniform(-4, 4)],
                [random.uniform(-6, 6), random.uniform(-5, 5)],
                [random.uniform(-6, 6), random.uniform(-5, 5)],
                [random.uniform(-4, 4), random.uniform(-4, 4)],
            ]
        )
        warp = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, warp, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    return out


def feature_tile(feature_map: np.ndarray, tile: int) -> np.ndarray:
    u8 = normalize_to_u8(feature_map)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return cv2.resize(color, (tile, tile), interpolation=cv2.INTER_NEAREST)


def weight_tile(kernel: np.ndarray, tile: int) -> np.ndarray:
    if kernel.ndim == 3 and kernel.shape[0] == 3:
        rgb = np.transpose(kernel, (1, 2, 0))
        u8 = normalize_to_u8(rgb)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.cvtColor(normalize_to_u8(kernel), cv2.COLOR_GRAY2BGR)
    return cv2.resize(bgr, (tile, tile), interpolation=cv2.INTER_NEAREST)


def draw_line_weight(canvas: np.ndarray, p0: Tuple[int, int], p1: Tuple[int, int], value: float, color: Tuple[int, int, int], thickness: int) -> None:
    cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)
    mx = int((p0[0] + p1[0]) * 0.5)
    my = int((p0[1] + p1[1]) * 0.5)
    label = f"{value:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(canvas, (mx - tw // 2 - 4, my - th - 4), (mx + tw // 2 + 4, my + 4), (255, 255, 255), -1)
    cv2.rectangle(canvas, (mx - tw // 2 - 4, my - th - 4), (mx + tw // 2 + 4, my + 4), (220, 220, 220), 1)
    put_text(canvas, label, (mx - tw // 2, my), scale=0.42, color=TITLE_BLUE, thickness=1)


def training_phase_label(trainer: "BlobTrainer") -> str:
    if trainer.awaiting_feedback:
        return "Waiting for keyboard feedback"
    if trainer.training_active:
        if trainer.epoch == 0 and trainer.step_in_epoch <= 2:
            return "Warming up the first filters"
        ratio = 0.0 if trainer.steps_per_epoch <= 0 else trainer.step_in_epoch / float(trainer.steps_per_epoch)
        if ratio < 0.25:
            return "Loading a new mini-batch"
        if ratio < 0.50:
            return "Comparing guesses with labels"
        if ratio < 0.75:
            return "Adjusting weights"
        return "Consolidating new patterns"
    if trainer.last_phase_label:
        return trainer.last_phase_label
    return "Ready to start"


class TinyBlobCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(24 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)
        self.reset_small()

    def reset_small(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=0.03)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x1 = F.relu(self.conv1(x))
        p1 = self.pool(x1)
        x2 = F.relu(self.conv2(p1))
        p2 = self.pool(x2)
        x3 = F.relu(self.conv3(p2))
        p3 = self.pool(x3)
        flat = torch.flatten(p3, 1)
        hidden = F.relu(self.fc1(flat))
        logits = self.fc2(hidden)
        if return_features:
            return logits, {"conv1": x1, "conv2": x2, "conv3": x3, "hidden": hidden}
        return logits


class BlobTrainer:
    def __init__(self) -> None:
        self.model = TinyBlobCNN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0016)
        self.positives: List[Sample] = []
        self.negatives: List[Sample] = []
        self.feedback_pool: List[Sample] = []
        self.augmentation_preview: List[np.ndarray] = []
        self.training_gallery: List[Tuple[np.ndarray, int]] = []

        self.training_active = False
        self.awaiting_feedback = False
        self.feedback_every = 10
        self.steps_per_epoch = 14
        self.batch_size = 8
        self.epoch = 0
        self.step_in_epoch = 0
        self.loss_ema = 0.0
        self.acc_ema = 0.0
        self.last_step_time = 0.0
        self.last_loss = 0.0
        self.last_grad_norm = 0.0
        self.last_message = "Collect a few object boxes to begin."
        self.last_feedback_kind = "neutral"
        self.last_phase_label = "Ready to start"
        self.backprop_pulse = 0.0
        self.last_best_box: Optional[Tuple[int, int, int, int]] = None
        self.detect_threshold = 0.78
        self.detect_margin = 0.12
        self.cached_scan: Optional[ScanResult] = None
        self.cached_preview = None
        self.preview_updated_at = 0.0
        self.treat_count = 0
        self.correction_count = 0
        self.recent_focus_label: Optional[int] = None

    def reset_project(self) -> None:
        self.model = TinyBlobCNN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0016)
        self.positives = []
        self.negatives = []
        self.feedback_pool = []
        self.augmentation_preview = []
        self.training_gallery = []
        self.training_active = False
        self.awaiting_feedback = False
        self.epoch = 0
        self.step_in_epoch = 0
        self.loss_ema = 0.0
        self.acc_ema = 0.0
        self.last_step_time = 0.0
        self.last_loss = 0.0
        self.last_grad_norm = 0.0
        self.last_feedback_kind = "neutral"
        self.last_phase_label = "Ready to start"
        self.backprop_pulse = 0.0
        self.last_best_box = None
        self.cached_scan = None
        self.cached_preview = None
        self.preview_updated_at = 0.0
        self.treat_count = 0
        self.correction_count = 0
        self.recent_focus_label = None
        self.last_message = "New project started. All samples and training state were cleared."

    def can_train(self) -> bool:
        return len(self.positives) >= 2 and len(self.negatives) >= 4

    def typical_box_size(self) -> int:
        sizes = [s.bbox_size for s in self.positives]
        if not sizes:
            return 180
        return int(np.median(sizes))

    def add_positive(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], source: str = "boxed") -> None:
        crop, side = make_square_crop(frame, bbox)
        self.positives.append(Sample(crop=crop.copy(), label=1, source=source, weight=1.0, bbox_size=side))
        mined = random_negative_crops(frame, bbox, side, count=6)
        for patch in mined:
            self.negatives.append(Sample(crop=patch.copy(), label=0, source="auto-negative", weight=1.0, bbox_size=side))
        preview = [jitter_crop(crop, heavy=True)]
        preview.extend(jitter_crop(patch, heavy=True) for patch in mined[:4])
        self.augmentation_preview = preview[:1]
        gallery_items = [(preview[0], 1)] + [(img, 0) for img in preview[1:10]]
        self.training_gallery = [(resize_square(img, 64), label) for img, label in gallery_items]
        self.cached_scan = None
        self.last_message = f"Captured labelled object #{len(self.positives)}. Added background examples automatically."

    def add_negative_crop(self, crop: np.ndarray, source: str = "manual-negative", weight: float = 1.0) -> None:
        side = max(crop.shape[:2]) if crop.size else 64
        self.negatives.append(Sample(crop=crop.copy(), label=0, source=source, weight=weight, bbox_size=side))
        preview = [jitter_crop(crop, heavy=True)]
        self.augmentation_preview = preview[:1]
        self.training_gallery = [(resize_square(img, 64), 0)]
        self.cached_scan = None
        self.last_message = f"Added a background example from {source}."

    def add_feedback(self, crop: np.ndarray, label: int) -> None:
        if crop is None or crop.size == 0:
            self.last_message = "No live crop available for feedback yet."
            return
        sample = Sample(crop=crop.copy(), label=label, source="kid-feedback", weight=3.0, bbox_size=max(crop.shape[:2]))
        self.feedback_pool.append(sample)
        if label == 1:
            self.positives.append(sample)
            self.treat_count += 1
            self.last_feedback_kind = "treat"
            self.last_message = "Feedback added: yes, this crop is the target object."
        else:
            self.negatives.append(sample)
            self.correction_count += 1
            self.last_feedback_kind = "correction"
            self.last_message = "Feedback added: no, this crop should count as background."
        was_training = self.training_active
        self.awaiting_feedback = False
        self.training_active = was_training and self.can_train()
        self.augmentation_preview = [jitter_crop(crop, heavy=True)]
        self.training_gallery = [(resize_square(self.augmentation_preview[0], 64), label)]
        self.backprop_pulse = 1.0
        self.recent_focus_label = label
        self.cached_scan = None

    def weighted_choice(self, samples: Sequence[Sample]) -> Sample:
        weights = [max(0.2, s.weight) for s in samples]
        return random.choices(list(samples), weights=weights, k=1)[0]

    def build_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Sample]:
        pos_count = max(1, int(round(self.batch_size * 0.4)))
        neg_count = max(1, self.batch_size - pos_count)
        pos_samples = [self.weighted_choice(self.positives) for _ in range(pos_count)]
        neg_samples = [self.weighted_choice(self.negatives) for _ in range(neg_count)]
        all_samples = pos_samples + neg_samples
        random.shuffle(all_samples)

        images = []
        labels = []
        weights = []
        gallery = []
        for sample in all_samples:
            aug = jitter_crop(sample.crop, heavy=True)
            images.append(crop_to_tensor(aug, size=64))
            labels.append(sample.label)
            weights.append(sample.weight)
            gallery.append((resize_square(aug, 64), sample.label))
        self.training_gallery = gallery[:10]
        batch_x = torch.stack(images, dim=0)
        batch_y = torch.tensor(labels, dtype=torch.long)
        batch_w = torch.tensor(weights, dtype=torch.float32)
        return batch_x, batch_y, batch_w, all_samples[0]

    def update_preview(self, crop: Optional[np.ndarray]) -> None:
        if crop is None or crop.size == 0:
            return
        if time.monotonic() - self.preview_updated_at < 0.12:
            return
        self.model.eval()
        with torch.no_grad():
            x = crop_to_tensor(crop, size=64).unsqueeze(0)
            logits, features = self.model(x, return_features=True)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        self.cached_preview = self.make_view_dict(crop, features, probs, label=None, loss=self.last_loss)
        self.preview_updated_at = time.monotonic()

    def make_view_dict(
        self,
        crop: np.ndarray,
        features: dict,
        probs: np.ndarray,
        label: Optional[int],
        loss: float,
    ) -> dict:
        def top_maps_with_idx(tensor: torch.Tensor, count: int) -> List[Tuple[int, np.ndarray]]:
            act = tensor[0].detach().cpu().numpy()
            means = act.mean(axis=(1, 2))
            order = np.argsort(means)[::-1][:count]
            return [(int(i), act[i]) for i in order]

        conv1_sel = top_maps_with_idx(features["conv1"], 4)
        conv2_sel = top_maps_with_idx(features["conv2"], 4)

        kernels1 = self.model.conv1.weight.detach().cpu().numpy()
        kernels2 = self.model.conv2.weight.detach().cpu().numpy()

        mix_weights = []
        for out_idx, _ in conv2_sel:
            row = []
            for in_idx, _ in conv1_sel:
                row.append(float(np.mean(np.abs(kernels2[out_idx, in_idx]))))
            row_sum = sum(row) or 1.0
            mix_weights.append([value / row_sum for value in row])

        fc_weights = np.abs(self.model.fc2.weight.detach().cpu().numpy())
        top_weights = []
        bottom_weights = []
        hidden_size = fc_weights.shape[1]
        segment = max(1, hidden_size // max(1, len(conv2_sel)))
        for idx in range(len(conv2_sel)):
            start = idx * segment
            end = hidden_size if idx == len(conv2_sel) - 1 else min(hidden_size, (idx + 1) * segment)
            top_weights.append(float(np.mean(fc_weights[1, start:end])))
            bottom_weights.append(float(np.mean(fc_weights[0, start:end])))
        top_sum = sum(top_weights) or 1.0
        bottom_sum = sum(bottom_weights) or 1.0
        top_weights = [value / top_sum for value in top_weights]
        bottom_weights = [value / bottom_sum for value in bottom_weights]

        return {
            "crop": resize_square(crop, 128),
            "conv1": [fmap for _, fmap in conv1_sel],
            "conv2": [fmap for _, fmap in conv2_sel],
            "mix_weights": mix_weights,
            "top_weights": top_weights,
            "bottom_weights": bottom_weights,
            "probs": probs,
            "label": label,
            "loss": loss,
        }

    def maybe_train_step(self) -> None:
        if not self.training_active or self.awaiting_feedback or not self.can_train():
            return
        now = time.monotonic()
        target_interval = 0.36 if self.epoch == 0 else 0.08
        if now - self.last_step_time < target_interval:
            return

        self.last_step_time = now
        self.model.train()
        batch_x, batch_y, batch_w, focus_sample = self.build_batch()
        logits, features = self.model(batch_x, return_features=True)
        losses = F.cross_entropy(logits, batch_y, reduction="none", label_smoothing=0.06)
        loss = torch.mean(losses * batch_w)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_sq = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_sq += float(torch.sum(param.grad.detach() ** 2))
        self.last_grad_norm = grad_sq ** 0.5
        self.optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            acc = float((preds == batch_y).float().mean().item())
            focus_probs = probs[0].detach().cpu().numpy()

        loss_val = float(loss.item())
        self.last_loss = loss_val
        self.loss_ema = loss_val if self.loss_ema == 0.0 else 0.9 * self.loss_ema + 0.1 * loss_val
        self.acc_ema = acc if self.acc_ema == 0.0 else 0.85 * self.acc_ema + 0.15 * acc
        self.step_in_epoch += 1
        self.backprop_pulse = 1.0
        if self.last_feedback_kind not in {"treat", "correction"}:
            self.last_feedback_kind = "training"

        self.cached_preview = self.make_view_dict(
            focus_sample.crop,
            {k: v[:1] for k, v in features.items()},
            focus_probs,
            label=int(batch_y[0].item()),
            loss=loss_val,
        )
        phase = training_phase_label(self)
        self.last_phase_label = phase
        self.last_message = f"Epoch {self.epoch + 1}: {phase.lower()}."
        print(
            f"[epoch {self.epoch + 1:03d} step {self.step_in_epoch:02d}/{self.steps_per_epoch}] "
            f"loss={loss_val:.4f} acc={acc * 100:.1f}% phase={phase}",
            flush=True,
        )

        if self.step_in_epoch >= self.steps_per_epoch:
            self.step_in_epoch = 0
            self.epoch += 1
            self.last_phase_label = "Finished an epoch"
            self.last_message = f"Finished epoch {self.epoch}."
            if self.epoch % self.feedback_every == 0:
                self.training_active = False
                self.awaiting_feedback = True
                self.last_message = (
                    f"Pause after {self.epoch} epochs. Show something to the camera and press T for yes or P for no."
                )
        self.cached_scan = None

    def proposal_boxes(self, frame_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
        h, w = frame_shape[:2]
        base = int(np.clip(self.typical_box_size(), 96, min(w, h) - 8))
        boxes: List[Tuple[int, int, int, int]] = []
        xs = np.linspace(base // 2, w - base // 2, 5, dtype=np.int32)
        ys = np.linspace(base // 2, h - base // 2, 3, dtype=np.int32)
        for cy in ys:
            for cx in xs:
                x0 = int(np.clip(cx - base // 2, 0, w - base))
                y0 = int(np.clip(cy - base // 2, 0, h - base))
                boxes.append((x0, y0, base, base))

        if self.last_best_box is not None:
            bx, by, bw, bh = self.last_best_box
            center_x = bx + bw // 2
            center_y = by + bh // 2
            for scale in (0.9, 1.0, 1.12):
                side = int(np.clip(base * scale, 88, min(w, h)))
                for dx in (-0.22, 0.0, 0.22):
                    for dy in (-0.22, 0.0, 0.22):
                        x0 = int(np.clip(center_x + dx * side - side / 2, 0, w - side))
                        y0 = int(np.clip(center_y + dy * side - side / 2, 0, h - side))
                        boxes.append((x0, y0, side, side))

        unique = []
        seen = set()
        for x, y, bw, bh in boxes:
            key = (int(round(x / 6.0)), int(round(y / 6.0)), int(round(bw / 6.0)))
            if key in seen:
                continue
            seen.add(key)
            unique.append((x, y, bw, bh))
        return unique

    def scan_frame(self, frame: np.ndarray) -> ScanResult:
        now = time.monotonic()
        if self.cached_scan is not None and now - self.cached_scan.checked_at < 0.12:
            return self.cached_scan

        boxes = self.proposal_boxes(frame.shape)
        crops = []
        valid_boxes = []
        for x, y, w, h in boxes:
            patch = frame[y : y + h, x : x + w]
            if patch.size == 0:
                continue
            crops.append(crop_to_tensor(patch, size=64))
            valid_boxes.append((x, y, w, h))

        if not crops:
            result = ScanResult([], [], None, None, 0.5, None)
            self.cached_scan = result
            return result

        self.model.eval()
        with torch.no_grad():
            batch = torch.stack(crops, dim=0)
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        best_idx = int(np.argmax(probs))
        best_box = valid_boxes[best_idx]
        bx, by, bw, bh = best_box
        best_crop = frame[by : by + bh, bx : bx + bw].copy()
        self.last_best_box = best_box
        second_best = float(np.partition(probs, -2)[-2]) if len(probs) > 1 else 0.0
        best_score = float(probs[best_idx])
        score_gap = best_score - second_best
        display_box = best_box if (best_score >= self.detect_threshold and score_gap >= self.detect_margin) else None

        result = ScanResult(
            boxes=valid_boxes,
            scores=[float(p) for p in probs.tolist()],
            best_box=best_box,
            display_box=display_box,
            best_score=best_score,
            best_crop=best_crop,
        )
        self.cached_scan = result
        return result


def open_camera(preferred_index: int = 0) -> cv2.VideoCapture:
    tried = [preferred_index] + [i for i in range(5) if i != preferred_index]
    for idx in tried:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
    raise RuntimeError("No camera found.")


def get_monitors() -> List[Monitor]:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    monitors: List[Monitor] = []
    callback_t = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.POINTER(RECT),
        ctypes.c_double,
    )

    def callback(_monitor, _hdc, rect_ptr, _data):
        rect = rect_ptr.contents
        monitors.append(Monitor(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top))
        return 1

    try:
        ctypes.windll.user32.EnumDisplayMonitors(0, 0, callback_t(callback), 0)
    except Exception:
        monitors = []

    if not monitors:
        return [Monitor(0, 0, 1600, 900)]
    monitors.sort(key=lambda item: (item.x, item.y))
    return monitors


def prepare_windows(monitors: List[Monitor]) -> Tuple[Monitor, Monitor]:
    primary = monitors[0]
    secondary = monitors[1] if len(monitors) > 1 else Monitor(primary.x + primary.w // 2, primary.y, primary.w // 2, primary.h)

    cv2.namedWindow(LIVE_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow(CNN_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.moveWindow(LIVE_WINDOW, primary.x, primary.y)
    cv2.resizeWindow(LIVE_WINDOW, primary.w, primary.h)
    cv2.moveWindow(CNN_WINDOW, secondary.x, secondary.y)
    cv2.resizeWindow(CNN_WINDOW, secondary.w, secondary.h)
    return primary, secondary


def guide_box(frame: np.ndarray, size: int) -> Tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    side = int(np.clip(size, 96, min(w, h) - 12))
    x = (w - side) // 2
    y = (h - side) // 2
    return x, y, side, side


def draw_confidence_bar(
    image: np.ndarray,
    rect: Rect,
    value: float,
    label: str,
    fill_color: Tuple[int, int, int],
) -> None:
    cv2.rectangle(image, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (255, 255, 255), 2)
    filled = int(rect.w * float(np.clip(value, 0.0, 1.0)))
    cv2.rectangle(image, (rect.x, rect.y), (rect.x + filled, rect.y + rect.h), fill_color, -1)
    put_text(image, label, (rect.x + 8, rect.y + rect.h - 10), scale=0.6, color=TEXT_DARK, thickness=2)


def draw_live_screen(frame: np.ndarray, trainer: BlobTrainer, scan: ScanResult) -> np.ndarray:
    out = frame.copy()

    if scan.display_box is not None and trainer.can_train():
        x, y, w, h = scan.display_box
        cv2.rectangle(out, (x, y), (x + w, y + h), NEUTRAL_BOX, 2)
        cv2.rectangle(out, (x, max(0, y - 30)), (x + 190, y), (255, 255, 255), -1)
        put_text(out, "Current focus", (x + 10, y - 9), scale=0.62, color=TITLE_BLUE, thickness=2)

    panel = Rect(18, 18, 390, 360)
    draw_panel(out, panel, title="")
    put_text(out, f"Labelled objects: {len(trainer.positives)}", (32, 58), scale=0.66, color=TEXT_DARK)
    put_text(out, f"Background examples: {len(trainer.negatives)}", (32, 88), scale=0.66, color=TEXT_DARK)
    put_text(out, f"Positive feedback: {trainer.treat_count}", (32, 118), scale=0.66, color=GOOD_GREEN)
    put_text(out, f"Negative feedback: {trainer.correction_count}", (32, 148), scale=0.66, color=BAD_RED)

    key_y = 196
    lines = [
        "C: label object",
        "X: label background example",
        "S: start or pause training",
        "T: yes, current crop is the object",
        "P: no, current crop is background",
        "R: reset project from scratch",
        "Q: quit",
    ]
    for i, text in enumerate(lines):
        put_text(out, text, (32, key_y + i * 28), scale=0.58, color=TEXT_DARK, thickness=1)

    return out


def draw_backprop_path(canvas: np.ndarray, start: Tuple[int, int], input_rect: Rect, padding: int = 24) -> None:
    bottom_center = (input_rect.x + input_rect.w // 2, input_rect.y + input_rect.h)
    right_x = canvas.shape[1] - padding
    bottom_y = canvas.shape[0] - padding
    route_points = [
        start,
        (right_x, start[1]),
        (right_x, bottom_y),
        (bottom_center[0], bottom_y),
        (bottom_center[0], bottom_center[1] + 16),
    ]
    for p0, p1 in zip(route_points[:-1], route_points[1:]):
        cv2.line(canvas, p0, p1, BACKPROP_RED, 3, cv2.LINE_AA)
    cv2.arrowedLine(canvas, route_points[-1], bottom_center, BACKPROP_RED, 3, tipLength=0.08)


def paste_thumbnail(canvas: np.ndarray, image: np.ndarray, rect: Rect, title: str = "") -> None:
    x0 = max(0, rect.x)
    y0 = max(0, rect.y)
    x1 = min(canvas.shape[1], rect.x + rect.w)
    y1 = min(canvas.shape[0], rect.y + rect.h)
    if x1 <= x0 or y1 <= y0:
        return

    if image is None or image.size == 0:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), GRID_BG, -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 2)
        return

    thumb = cv2.resize(image, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    canvas[y0:y1, x0:x1] = thumb
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 2)
    if title:
        put_text(canvas, title, (x0 + 8, max(20, y0 - 10)), scale=0.56, color=TITLE_BLUE)


def draw_connection_grid(
    canvas: np.ndarray,
    basic_rects: List[Rect],
    deep_rects: List[Rect],
    weights: List[List[float]],
) -> None:
    if not basic_rects or not deep_rects or not weights:
        return
    for deep_idx, deep_rect in enumerate(deep_rects):
        row = weights[deep_idx]
        max_w = max(max(row), 1e-6)
        for basic_idx, basic_rect in enumerate(basic_rects):
            value = float(row[basic_idx])
            thickness = 1 + int(4 * (value / max_w))
            p0 = (basic_rect.x + basic_rect.w, basic_rect.y + basic_rect.h // 2)
            p1 = (deep_rect.x, deep_rect.y + deep_rect.h // 2)
            draw_line_weight(canvas, p0, p1, value, LINK_BLUE, thickness)


def draw_output_links(
    canvas: np.ndarray,
    deep_rects: List[Rect],
    out_top: Rect,
    out_bottom: Rect,
    top_weights: List[float],
    bottom_weights: List[float],
) -> None:
    if not deep_rects or not top_weights or not bottom_weights:
        return
    max_top = max(max(top_weights), 1e-6)
    max_bottom = max(max(bottom_weights), 1e-6)
    for rect, top_value, bottom_value in zip(deep_rects, top_weights, bottom_weights):
        start = (rect.x + rect.w, rect.y + rect.h // 2)
        draw_line_weight(canvas, start, (out_top.x - 8, out_top.y + 44), float(top_value), ACCENT_ORANGE, 1 + int(4 * (top_value / max_top)))
        draw_line_weight(canvas, start, (out_bottom.x - 8, out_bottom.y + 44), float(bottom_value), PURPLE, 1 + int(4 * (bottom_value / max_bottom)))


def draw_network_screen(
    trainer: BlobTrainer,
    live_crop: Optional[np.ndarray],
    screen_rect: Monitor,
) -> np.ndarray:
    canvas = np.full((screen_rect.h, screen_rect.w, 3), SOFT_BG, dtype=np.uint8)
    view = trainer.cached_preview
    if view is None and live_crop is not None:
        trainer.update_preview(live_crop)
        view = trainer.cached_preview

    cw = canvas.shape[1]
    ch = canvas.shape[0]
    tile = max(72, min(120, int(ch * 0.12)))

    put_text(canvas, "CNNs", (24, 52), scale=1.45, color=TITLE_BLUE, thickness=3)
    put_text(canvas, "Convolutional Neural Networks", (24, 92), scale=0.95, color=TITLE_BLUE, thickness=2)
    put_text(canvas, "Live training visualisation", (24, 124), scale=0.68, color=ACCENT_ORANGE, thickness=2)

    input_rect = Rect(24, int(ch * 0.23), int(cw * 0.22), int(ch * 0.44))
    draw_panel(canvas, input_rect, title="Camera Crop")
    if view is not None:
        paste_thumbnail(canvas, view["crop"], Rect(input_rect.x + 14, input_rect.y + 40, input_rect.w - 28, input_rect.h - 54))
    else:
        draw_multiline(
            canvas,
            "No crop yet. Capture a few boxes first, then the live camera crop will appear here.",
            (input_rect.x + 16, input_rect.y + 80),
            width=24,
            scale=0.72,
            color=TEXT_DARK,
        )

    sample_rect = Rect(input_rect.x + input_rect.w + 16, input_rect.y, max(160, int(cw * 0.10)), input_rect.h)
    draw_panel(canvas, sample_rect, title="Training samples")
    sample_gap = 8
    sample_tile_w = max(46, min(78, (sample_rect.w - 16 - sample_gap) // 2))
    sample_tile_h = max(44, min(68, (sample_rect.h - 58 - 4 * sample_gap) // 5))
    gallery = trainer.training_gallery[:10]
    for idx in range(10):
        x = sample_rect.x + 8 + (idx % 2) * (sample_tile_w + sample_gap)
        y = sample_rect.y + 40 + (idx // 2) * (sample_tile_h + sample_gap)
        if idx < len(gallery):
            thumb, label = gallery[idx]
            paste_thumbnail(canvas, resize_square(thumb, 64), Rect(x, y, sample_tile_w, sample_tile_h))
            border = GOOD_GREEN if label == 1 else BAD_RED
            cv2.rectangle(canvas, (x, y), (x + sample_tile_w, y + sample_tile_h), border, 3)
        else:
            cv2.rectangle(canvas, (x, y), (x + sample_tile_w, y + sample_tile_h), GRID_BG, -1)
            cv2.rectangle(canvas, (x, y), (x + sample_tile_w, y + sample_tile_h), (255, 255, 255), 1)

    col1_x = int(cw * 0.40)
    col2_x = int(cw * 0.59)
    out_x = int(cw * 0.82)
    ys = [int(ch * 0.22), int(ch * 0.36), int(ch * 0.50), int(ch * 0.64)]

    put_text(canvas, "Basic Features", (col1_x - 4, 170), scale=0.72, color=TITLE_BLUE)
    put_text(canvas, "Deep Features", (col2_x - 4, 170), scale=0.72, color=TITLE_BLUE)
    put_text(canvas, "Outputs", (out_x - 10, 170), scale=0.72, color=TITLE_BLUE)

    basic_labels = ["Edges", "Corners", "Contrast", "Texture"]
    deep_labels = ["Shape mix", "Part layout", "Repeated cues", "Object hint"]
    basic_rects: List[Rect] = []
    deep_rects: List[Rect] = []

    if view is not None:
        for idx, fmap in enumerate(view["conv1"]):
            tile_img = feature_tile(fmap, tile)
            rect = Rect(col1_x, ys[idx], tile, tile)
            basic_rects.append(rect)
            paste_thumbnail(canvas, tile_img, rect)
            put_text(canvas, basic_labels[idx], (rect.x, rect.y + rect.h + 18), scale=0.50, color=TEXT_DARK, thickness=1)

        for idx, fmap in enumerate(view["conv2"]):
            tile_img = feature_tile(fmap, tile)
            rect = Rect(col2_x, ys[idx], tile, tile)
            deep_rects.append(rect)
            paste_thumbnail(canvas, tile_img, rect)
            put_text(canvas, deep_labels[idx], (rect.x, rect.y + rect.h + 18), scale=0.50, color=TEXT_DARK, thickness=1)

        draw_connection_grid(canvas, basic_rects, deep_rects, view["mix_weights"])

        probs = view["probs"]
        object_prob = float(probs[1])
        other_prob = float(probs[0])
        out_top = Rect(out_x, int(ch * 0.30), int(cw * 0.13), 88)
        out_bottom = Rect(out_x, int(ch * 0.53), int(cw * 0.13), 88)
        draw_panel(canvas, out_top, title="")
        draw_panel(canvas, out_bottom, title="")
        put_text(canvas, "Object", (out_top.x + 12, out_top.y + 30), scale=0.62, color=TITLE_BLUE)
        put_text(canvas, "Background", (out_bottom.x + 12, out_bottom.y + 30), scale=0.62, color=TITLE_BLUE)
        draw_confidence_bar(canvas, Rect(out_top.x + 12, out_top.y + 45, out_top.w - 24, 24), object_prob, f"{object_prob * 100:.0f}%", GOOD_GREEN)
        draw_confidence_bar(canvas, Rect(out_bottom.x + 12, out_bottom.y + 45, out_bottom.w - 24, 24), other_prob, f"{other_prob * 100:.0f}%", BAD_RED)
        draw_output_links(canvas, deep_rects, out_top, out_bottom, view["top_weights"], view["bottom_weights"])

    metrics_rect = Rect(int(cw * 0.72), 20, int(cw * 0.26), 118)
    draw_panel(canvas, metrics_rect, title=f"Training state (Epoch {trainer.epoch})")
    put_text(canvas, training_phase_label(trainer), (metrics_rect.x + 14, metrics_rect.y + 46), scale=0.58, color=TEXT_DARK, thickness=1)
    put_text(canvas, f"Batch accuracy: {trainer.acc_ema * 100:.0f}%", (metrics_rect.x + 14, metrics_rect.y + 76), scale=0.58, color=TEXT_DARK, thickness=1)
    if trainer.awaiting_feedback:
        put_text(canvas, "Press T or P", (metrics_rect.x + 14, metrics_rect.y + 104), scale=0.58, color=BAD_RED, thickness=1)
    elif trainer.training_active:
        put_text(canvas, "Training running", (metrics_rect.x + 14, metrics_rect.y + 104), scale=0.58, color=GOOD_GREEN, thickness=1)
    elif trainer.epoch > 0 or trainer.step_in_epoch > 0:
        put_text(canvas, "Press S to continue", (metrics_rect.x + 14, metrics_rect.y + 104), scale=0.58, color=TITLE_BLUE, thickness=1)
    else:
        put_text(canvas, "Press S to start", (metrics_rect.x + 14, metrics_rect.y + 104), scale=0.58, color=TITLE_BLUE, thickness=1)

    draw_backprop_path(canvas, start=(out_x + 28, int(ch * 0.48)), input_rect=input_rect, padding=26)
    put_text(canvas, "Back Propagation", (int(cw * 0.63), int(ch * 0.96)), scale=0.78, color=TITLE_BLUE, thickness=2)

    return canvas


def select_box(frame: np.ndarray, prompt: str) -> Optional[Tuple[int, int, int, int]]:
    frozen = frame.copy()
    cv2.rectangle(frozen, (0, 0), (frozen.shape[1], 48), (255, 255, 255), -1)
    put_text(frozen, prompt, (16, 32), scale=0.8, color=TITLE_BLUE, thickness=2)
    roi = cv2.selectROI(ANNOTATE_WINDOW, frozen, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(ANNOTATE_WINDOW)
    x, y, w, h = map(int, roi)
    if w <= 5 or h <= 5:
        return None
    return x, y, w, h


def current_feedback_crop(scan: ScanResult, frame: np.ndarray, trainer: BlobTrainer) -> np.ndarray:
    if scan.best_crop is not None and scan.best_crop.size > 0:
        return scan.best_crop
    gx, gy, gw, gh = guide_box(frame, trainer.typical_box_size())
    return frame[gy : gy + gh, gx : gx + gw].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Teach how a CNN learns from object and background examples.")
    parser.add_argument("--camera-index", type=int, default=0, help="Preferred webcam index.")
    args = parser.parse_args()

    cap = open_camera(args.camera_index)
    monitors = get_monitors()
    _, cnn_monitor = prepare_windows(monitors)
    trainer = BlobTrainer()

    last_frame = None

    try:
        while True:
            ok, frame = cap.read()
            if ok:
                frame = cv2.flip(frame, 1)
                last_frame = frame.copy()
            elif last_frame is not None:
                frame = last_frame.copy()
            else:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            scan = trainer.scan_frame(frame)
            live_crop = current_feedback_crop(scan, frame, trainer)
            trainer.update_preview(live_crop)
            trainer.maybe_train_step()

            live_screen = draw_live_screen(frame, trainer, scan)
            cnn_screen = draw_network_screen(trainer, live_crop, cnn_monitor)

            cv2.imshow(LIVE_WINDOW, live_screen)
            cv2.imshow(CNN_WINDOW, cnn_screen)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and last_frame is not None:
                bbox = select_box(last_frame, "Draw a box around the object, then press SPACE or ENTER.")
                if bbox is not None:
                    trainer.add_positive(last_frame, bbox)
            elif key == ord("x") and last_frame is not None:
                bbox = select_box(last_frame, "Draw a background box: something that is not the target object.")
                if bbox is not None:
                    crop, _ = make_square_crop(last_frame, bbox)
                    trainer.add_negative_crop(crop, source="boxed-negative", weight=2.0)
            elif key == ord("s"):
                if trainer.can_train():
                    trainer.training_active = not trainer.training_active
                    trainer.awaiting_feedback = False
                    trainer.last_feedback_kind = "training"
                    trainer.last_message = "Training running." if trainer.training_active else f"Paused at epoch {trainer.epoch}."
                else:
                    trainer.last_message = "Need at least 2 labelled object boxes before training can start."
            elif key == ord("r"):
                trainer.reset_project()
            elif key == ord("t"):
                crop = current_feedback_crop(scan, frame, trainer)
                trainer.add_feedback(crop, label=1)
            elif key == ord("p"):
                crop = current_feedback_crop(scan, frame, trainer)
                trainer.add_feedback(crop, label=0)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
