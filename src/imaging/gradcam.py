"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Medical Image Interpretability.
Follows Section 2.3:
- Hooks final convolutional layer of backbone (e.g. ResNet-18 layer4).
- Computes target class gradients w.r.t feature maps.
- Pools channel gradients to compute importance weights.
- Generates heatmap and blends with original clinical image.
- Returns base64 encoded PNG or image array.
"""

import io
import base64
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_colormap_jet(heatmap: np.ndarray) -> np.ndarray:
    """
    Applies custom jet/turbo-like colormap without requiring external OpenCV.
    Heatmap: 2D float array in [0, 1]. Returns RGB uint8 array (H, W, 3).
    """
    x = np.clip(heatmap, 0.0, 1.0)
    # Simple RGB piece-wise color interpolation (blue -> cyan -> yellow -> red)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


class GradCAM:
    """
    Computes Grad-CAM saliency heatmaps for PyTorch vision models.
    """
    def __init__(self, model: nn.Module, target_layer_name: str = "layer4"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break

        if target_layer is None:
            # Fallback to last conv/sequential child
            children = list(self.model.children())
            target_layer = children[-2] if len(children) >= 2 else children[-1]

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generates 2D Grad-CAM heatmap normalized to [0, 1].
        input_tensor: 1 x C x H x W tensor.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Return uniform if hooks didn't capture
            _, _, h, w = input_tensor.shape
            return np.zeros((h, w), dtype=np.float32)

        # Global average pool gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input tensor resolution
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def generate_overlay(self, original_img: Image.Image,
                         input_tensor: torch.Tensor,
                         class_idx: Optional[int] = None,
                         alpha: float = 0.5) -> Tuple[Image.Image, str]:
        """
        Blends Grad-CAM heatmap onto the original PIL image.
        Returns blended PIL Image and base64 PNG data URL string.
        """
        heatmap = self.generate_heatmap(input_tensor, class_idx=class_idx)
        w, h = original_img.size

        # Resize heatmap to match original image dimensions
        pil_heat = Image.fromarray((heatmap * 255.0).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        heat_arr = np.array(pil_heat) / 255.0

        heat_colored = apply_colormap_jet(heat_arr)
        orig_arr = np.array(original_img.convert('RGB'))

        # Blend
        blended = (orig_arr * (1.0 - alpha) + heat_colored * alpha).astype(np.uint8)
        blended_pil = Image.fromarray(blended)

        # Export to base64
        buf = io.BytesIO()
        blended_pil.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        data_url = f"data:image/png;base64,{b64_str}"

        return blended_pil, data_url
