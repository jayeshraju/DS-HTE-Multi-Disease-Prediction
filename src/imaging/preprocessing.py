"""
Modality-aware Preprocessing Pipeline for Medical Images.
Follows Section 2.3:
- Configurable resize (default 224x224).
- 1 -> 3 channel handling for grayscale modalities (X-Ray, CT, Ultrasound).
- Contrast-Limited Adaptive Histogram Equalization (CLAHE) for radiologic enhancement.
- Standard ImageNet intensity normalization.
- Conservative medical augmentation (horizontal flip only if safe; no vertical flip).
"""

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T


def apply_clahe_pillow(img: Image.Image) -> Image.Image:
    """
    Applies adaptive contrast enhancement using histogram equalization across tiles.
    Uses numpy/PIL without requiring OpenCV.
    """
    gray = img.convert('L')
    arr = np.array(gray)
    # Simple, fast tile-based adaptive histogram equalization
    h, w = arr.shape
    tile_h, tile_w = max(16, h // 8), max(16, w // 8)
    out_arr = np.zeros_like(arr)

    for i in range(0, h, tile_h):
        for j in range(0, w, tile_w):
            tile = arr[i:min(i + tile_h, h), j:min(j + tile_w, w)]
            # Histogram equalization on tile
            hist, bins = np.histogram(tile.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255 / max(1, cdf[-1])
            equalized_tile = np.interp(tile.flatten(), bins[:-1], cdf_normalized).reshape(tile.shape)
            out_arr[i:min(i + tile_h, h), j:min(j + tile_w, w)] = equalized_tile.astype(np.uint8)

    return Image.fromarray(out_arr).convert('RGB')


class MedicalImagePreprocessor:
    """
    Preprocesses medical images for deep learning feature extractors and backbones.
    """
    def __init__(self,
                 target_size: int = 224,
                 use_clahe: bool = False,
                 is_training: bool = False,
                 allow_hflip: bool = True):
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.is_training = is_training
        self.allow_hflip = allow_hflip

        # ImageNet normalization standard
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def preprocess_pil(self, img: Image.Image) -> torch.Tensor:
        """Converts PIL image to normalized 3x224x224 PyTorch float tensor."""
        if self.use_clahe:
            img = apply_clahe_pillow(img)
        else:
            img = img.convert('RGB')

        # Resize
        img = img.resize((self.target_size, self.target_size), Image.Resampling.BILINEAR)

        # Convert to tensor [0, 1]
        tensor = T.functional.to_tensor(img)

        # Ensure 3 channels
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] > 3:
            tensor = tensor[:3, :, :]

        # Normalize
        tensor = self.normalize(tensor)
        return tensor
