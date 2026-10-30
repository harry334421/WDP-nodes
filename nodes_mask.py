"""WDP-nodes — mask utilities for ComfyUI.

Currently ships one node:

    WDP Load Mask Image  (WDP_LoadMaskImage)
        Upload your own black/white mask picture (PNG / JPG / WebP / ...) and
        convert it into a ComfyUI MASK tensor.  Optional invert, binarize
        (threshold) and resize make it usable without any extra preprocessing.

Convention: in ComfyUI a MASK value of 1.0 = "area to change" (white).
"""

from __future__ import annotations

import hashlib
import logging
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths

LOGGER = logging.getLogger(__name__)

CHANNEL_OPTIONS = ["luminance", "red", "green", "blue", "alpha", "average", "max"]
RESIZE_MODES = ["none", "stretch", "fit"]

# Only used when folder_paths.filter_files_content_types is unavailable.
_FALLBACK_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".tif", ".tiff", ".gif", ".avif",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def list_input_images() -> list[str]:
    """Every image file directly inside ComfyUI's ``input`` directory."""
    try:
        input_dir = folder_paths.get_input_directory()
    except Exception:  # pragma: no cover - folder_paths always provides this
        return []

    if not os.path.isdir(input_dir):
        return []

    files = [
        name
        for name in os.listdir(input_dir)
        if not name.startswith(".") and os.path.isfile(os.path.join(input_dir, name))
    ]

    try:
        files = folder_paths.filter_files_content_types(files, ["image"])
    except Exception:
        files = [f for f in files if os.path.splitext(f)[1].lower() in _FALLBACK_IMAGE_EXTS]

    return sorted(files, key=str.lower)


def _frame_to_mask(frame: Image.Image, channel: str) -> np.ndarray:
    """Convert one PIL frame into a float32 grayscale array in [0, 1]."""
    had_alpha = "A" in frame.getbands() or (
        frame.mode == "P" and "transparency" in frame.info
    )

    rgba = np.asarray(frame.convert("RGBA"), dtype=np.float32) / 255.0
    r, g, b, a = rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3]

    if channel == "red":
        mask = r
    elif channel == "green":
        mask = g
    elif channel == "blue":
        mask = b
    elif channel == "alpha":
        if not had_alpha:
            LOGGER.warning(
                "WDP Load Mask Image: image has no alpha channel, "
                "falling back to luminance for channel='alpha'."
            )
        mask = a
    elif channel == "average":
        mask = (r + g + b) / 3.0
    elif channel == "max":
        mask = np.maximum(np.maximum(r, g), b)
    else:  # "luminance"
        mask = 0.299 * r + 0.587 * g + 0.114 * b

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _resize_mask(mask: np.ndarray, mode: str, width: int, height: int) -> np.ndarray:
    """Resize a float mask to ``width`` x ``height``; mode is stretch or fit."""
    src_h, src_w = mask.shape
    if (src_w, src_h) == (width, height):
        return mask

    # "F" mode keeps the float precision, so soft / anti-aliased edges survive.
    src = Image.fromarray(mask, mode="F")

    if mode == "stretch":
        resized = src.resize((width, height), Image.BILINEAR)
        return np.asarray(resized, dtype=np.float32)

    # fit: keep aspect ratio, pad with 0 (black = not selected).
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = src.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("F", (width, height), 0.0)
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return np.asarray(canvas, dtype=np.float32)


def _latent_target_size(latent) -> tuple[int, int] | None:
    """Pixel size implied by a LATENT (latent H/W * 8), or None."""
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if torch.is_tensor(samples) and samples.ndim >= 4:
        return int(samples.shape[3]) * 8, int(samples.shape[2]) * 8
    return None


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------

class WDP_LoadMaskImage:
    """Load a black/white picture and turn it into a MASK."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    list_input_images(),
                    {
                        "image_upload": True,
                        "tooltip": "上传自己的遮罩图（推荐纯黑白 PNG）。白色 = 选中/重绘，黑色 = 保留。",
                    },
                ),
                "channel": (
                    CHANNEL_OPTIONS,
                    {
                        "default": "luminance",
                        "tooltip": "取哪一路作为遮罩。纯黑白灰度图用 luminance 即可；"
                                   "带透明通道的图可选 alpha。",
                    },
                ),
                "invert": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "反转（黑色=选中）",
                        "label_off": "不反转（白色=选中）",
                        "tooltip": "如果遮罩图是白底黑图，开启反转。",
                    },
                ),
                "binarize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "二值化",
                        "label_off": "保留灰度",
                        "tooltip": "按阈值把遮罩变成纯 0/1，消除 JPG 压缩噪点和抗锯齿灰边。",
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "只在开启二值化时生效：像素值 >= 阈值算白色（选中）。",
                    },
                ),
                "resize_mode": (
                    RESIZE_MODES,
                    {
                        "default": "stretch",
                        "tooltip": "stretch 拉伸到目标尺寸；fit 等比缩放后居中补黑边；"
                                   "none 保持原图尺寸。目标尺寸优先取 reference_image，"
                                   "其次是 latent，最后是 width/height。",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "手动的目标宽度（没有连接 reference_image / latent 时使用）。"
                                   "0 = 保持原尺寸。",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "手动的目标高度。0 = 保持原尺寸。",
                    },
                ),
            },
            "optional": {
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": "可选：连接要重绘的图像，遮罩自动缩放到该图像尺寸（优先级最高）。",
                    },
                ),
                "latent": (
                    "LATENT",
                    {
                        "tooltip": "可选：连接当前 latent，遮罩自动缩放到 latent 尺寸×8"
                                   "（像素尺寸），用于直接替换 LatentToMaskBB。"
                                   "resize_mode 为 none 时不缩放。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "load_mask"
    CATEGORY = "WDP-nodes/遮罩"

    DESCRIPTION = (
        "上传自己的黑白遮罩图并转换为 MASK：白色=1（重绘区域），黑色=0（保留区域）。"
        "支持任意形状，可替代 ComfyUI-ppm 的 LatentToMaskBB（矩形）。"
        "支持反转、二值化、缩放，可连接 reference_image / latent 自动匹配尺寸。"
    )
    SEARCH_ALIASES = [
        "mask", "load mask", "upload mask", "mask image", "inpaint mask",
        "遮罩", "蒙版", "上传遮罩", "加载遮罩", "黑白遮罩",
    ]

    def load_mask(
        self,
        image: str,
        channel: str,
        invert: bool,
        binarize: bool,
        threshold: float,
        resize_mode: str,
        width: int,
        height: int,
        reference_image=None,
        latent=None,
    ):
        if not folder_paths.exists_annotated_filepath(image):
            raise FileNotFoundError(f"Mask image not found: {image}")

        image_path = folder_paths.get_annotated_filepath(image)

        # Target size: reference_image > latent > manual width/height.
        target_size = None
        if resize_mode != "none":
            if torch.is_tensor(reference_image) and reference_image.ndim >= 3:
                target_size = (int(reference_image.shape[2]), int(reference_image.shape[1]))
            elif latent is not None:
                target_size = _latent_target_size(latent)

            if target_size is None and width > 0 and height > 0:
                target_size = (int(width), int(height))

            if target_size is None:
                LOGGER.debug(
                    "WDP Load Mask Image: no target size available "
                    "(reference_image/latent unconnected and width/height are 0); "
                    "keeping the original size.",
                )
        elif latent is not None or reference_image is not None:
            LOGGER.info(
                "WDP Load Mask Image: reference_image/latent is connected but "
                "resize_mode is 'none', so the mask keeps its original size.",
            )

        masks: list[torch.Tensor] = []
        previews: list[torch.Tensor] = []
        expected_size = None

        with Image.open(image_path) as opened:
            opened = ImageOps.exif_transpose(opened)
            for frame in ImageSequence.Iterator(opened):
                frame = ImageOps.exif_transpose(frame)

                if expected_size is None:
                    expected_size = frame.size
                elif frame.size != expected_size:
                    # Animated files with mixed frame sizes cannot share one batch.
                    continue

                mask = _frame_to_mask(frame, channel)

                if invert:
                    mask = 1.0 - mask

                if target_size is not None:
                    mask = _resize_mask(mask, resize_mode, target_size[0], target_size[1])

                if binarize:
                    mask = (mask >= threshold).astype(np.float32)

                mask = np.clip(mask, 0.0, 1.0)
                masks.append(torch.from_numpy(mask)[None, ...])
                preview = np.repeat(mask[..., None], 3, axis=2)
                previews.append(torch.from_numpy(preview)[None, ...])

        if not masks:
            raise ValueError(f"Could not read any frame from mask image: {image}")

        return torch.cat(masks, dim=0), torch.cat(previews, dim=0)

    @classmethod
    def IS_CHANGED(cls, image: str, **kwargs):
        try:
            return _file_sha256(folder_paths.get_annotated_filepath(image))
        except Exception:
            return ""

    @classmethod
    def VALIDATE_INPUTS(cls, image: str, **kwargs):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid mask image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {
    "WDP_LoadMaskImage": WDP_LoadMaskImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WDP_LoadMaskImage": "WDP 遮罩图上传 (Load Mask Image)",
}
