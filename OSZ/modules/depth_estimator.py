"""Monocular depth estimator wrapper — MiDaS v2.1 Small (MiDaSNet-small).

Pipeline
--------
1. Load a locally cloned MiDaS repo (``cfg.MIDAS_REPO_PATH``) via
   ``torch.hub.load(..., source='local')`` and a LOCAL checkpoint
   (``cfg.MIDAS_MODEL_PATH``). Nothing is ever downloaded at runtime.
2. Predict inverse-like relative depth (MiDaS convention: larger = closer).
3. ``align_to_lidar`` auto-detects the inverse family and fits
   ``metric = scale / rel + shift``, reusing the existing LiDAR-alignment code.

Notes
-----
- The small model entry in the repo's ``hubconf.py`` is ``MiDaS_small``
  (EfficientNet-Lite3 encoder, 256x256 input) — ``DPT_Small`` does not
  exist there (verified against upstream master). Preprocessing uses the
  official ``small_transform`` from the same hubconf, so the model input
  distribution matches training exactly.
- MiDaS runs on torch 1.9.1 and needs NO timm for the small model, which is
  why it is the default depth model for the single-environment setup.
- If the repo or checkpoint is missing, :meth:`_load` raises; the pipeline's
  ``get_estimator`` catches that and falls back to :class:`MockDepthEstimator`
  (LiDAR densified depth) so the rest of the pipeline still runs.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from OSZ import config as cfg


class DepthEstimator:
    """MiDaS v2.1 Small (MiDaSNet-small) depth estimator (local weights only).

    Parameters
    ----------
    model_path : str, optional
        Local MiDaS checkpoint path. Defaults to ``cfg.MIDAS_MODEL_PATH``.
    repo_path : str, optional
        Local MiDaS repository path (cloned once, must contain
        ``hubconf.py``). Defaults to ``cfg.MIDAS_REPO_PATH``.
    device : str, optional
        ``'cpu'`` or ``'cuda'``. Auto-detected if ``None``.

    Examples
    --------
    >>> est = DepthEstimator()
    >>> depth_metric = est.infer(image, lidar_sparse_depth=sparse_depth)
    """

    def __init__(
        self,
        model_path: str = cfg.MIDAS_MODEL_PATH,
        repo_path: str = cfg.MIDAS_REPO_PATH,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.repo_path = repo_path
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self._model = None
        self._transform = None

    @staticmethod
    def _cuda_available() -> bool:
        """Check whether PyTorch CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _load(self):
        """Lazily build the MiDaS model and its preprocessing transform."""
        if self._model is not None:
            return self._model

        import torch
        from torch.hub import load as hub_load

        repo_dir = Path(self.repo_path)
        if not (repo_dir / "hubconf.py").exists():
            raise FileNotFoundError(
                f"MiDaS repo not found at {repo_dir}. Clone it once:\n"
                f"  git clone https://github.com/isl-org/MiDaS.git {repo_dir}"
            )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"MiDaS checkpoint not found at {self.model_path}.\n"
                f"Download it once (see REPRODUCE.md 3.3):\n  {cfg.MIDAS_MODEL_URL}"
            )

        print(
            f"[DepthEstimator] loading MiDaS v2.1 Small (MiDaSNet-small) "
            f"from {self.model_path} on {self.device} ..."
        )
        # Official repo code is used verbatim via torch.hub (source='local'),
        # so the checkpoint keys always match the model definition.
        # NOTE: the small model entry is `MiDaS_small` (EfficientNet-Lite3
        # encoder, no timm needed) — `DPT_Small` does NOT exist in the repo's
        # hubconf.py (verified against master branch).
        sys.path.insert(0, str(repo_dir))
        model = hub_load(str(repo_dir), "MiDaS_small", pretrained=False, source="local")
        ckpt = torch.load(self.model_path, map_location="cpu")
        try:
            model.load_state_dict(ckpt, strict=True)
        except RuntimeError as e:
            # Tolerate a "model."/"module." prefix (checkpoint saved under a
            # wrapping module) by stripping it; anything else fails loudly.
            print(
                f"[DepthEstimator] strict load failed ({type(e).__name__}); "
                "retrying with 'model.'/'module.' prefix stripped"
            )
            stripped = {}
            for k, v in ckpt.items():
                if k.startswith("module."):
                    stripped[k[7:]] = v
                elif k.startswith("model."):
                    stripped[k[6:]] = v
                else:
                    stripped[k] = v
            model.load_state_dict(stripped, strict=True)
        model.eval()
        model.to(self.device)
        self._model = model

        # Official MiDaS preprocessing for the small model: the repo's
        # hubconf `transforms()` exposes `small_transform` (256x256,
        # ImageNet normalization, keep_aspect_ratio) — use it verbatim so the
        # model input distribution matches training exactly.
        midas_transforms = hub_load(str(repo_dir), "transforms", source="local")
        self._transform = midas_transforms.small_transform
        return self._model

    def infer_relative(
        self,
        image: np.ndarray,
        target_size: Optional[tuple] = None,
    ) -> np.ndarray:
        """Predict MiDaS inverse depth (larger = closer) for one RGB image.

        Parameters
        ----------
        image : np.ndarray
            (H, W, 3) uint8 RGB image.
        target_size : tuple, optional
            ``(H, W)`` to resize the output. If ``None``, the input image size
            is used (the 256x256 prediction is interpolated back).

        Returns
        -------
        np.ndarray
            (H, W) float32 relative depth map (inverse-like).
        """
        import torch
        import torch.nn.functional as F

        model = self._load()
        # `small_transform` already returns a (1, 3, 256, 256) CPU tensor.
        # Its Compose starts with `lambda img: {"image": img / 255.0}` which
        # expects the raw ndarray, NOT a {"image": ...} dict.
        input_t = self._transform(image).to(self.device)
        with torch.no_grad():
            pred = model(input_t)  # (1, 1, 256, 256)
        depth = F.interpolate(
            pred,
            size=(image.shape[0], image.shape[1]),
            mode="bicubic",
            align_corners=False,
        )[0, 0].cpu().numpy().astype(np.float32)

        if target_size is not None:
            from PIL import Image as PILImage
            depth = np.array(
                PILImage.fromarray(depth).resize(
                    (target_size[1], target_size[0]), PILImage.BILINEAR
                ),
                dtype=np.float32,
            )
        return depth

    @staticmethod
    def align_to_lidar(
        depth_rel: np.ndarray,
        lidar_sparse: np.ndarray,
        min_points: int = cfg.MIN_ALIGN_POINTS,
        max_metric_depth: float = cfg.MAX_METRIC_DEPTH_M,
    ) -> np.ndarray:
        """Fit scale and shift to convert relative depth to metric depth.

        Solves ``metric = scale * rel + shift`` via least squares on LiDAR
        pixels. For inverse-like relative depth (MiDaS convention: larger rel
        = closer), ``metric = scale / rel + shift`` is also tried.

        Robustness additions:

        - If the fitted scale is degenerate (``|scale|`` below a threshold),
          fall back to a median-ratio estimate so depth still varies with the
          prediction.
        - Clip the output to ``[0, max_metric_depth]`` to avoid unrealistic
          far walls.

        Parameters
        ----------
        depth_rel : np.ndarray
            (H, W) relative depth map.
        lidar_sparse : np.ndarray
            (H, W) sparse LiDAR depth in metres (0 = invalid).
        min_points : int, optional
            Minimum LiDAR points required for least-squares fitting. Falls
            back to median-ratio scaling if fewer points are available.
        max_metric_depth : float, optional
            Maximum allowable output depth in metres.

        Returns
        -------
        np.ndarray
            (H, W) float32 metric depth map.
        """
        valid = lidar_sparse > 0
        n_valid = valid.sum()
        if n_valid == 0:
            raise ValueError("No valid LiDAR depth for alignment.")

        rel_vals = depth_rel[valid]
        lidar_vals = lidar_sparse[valid]

        # Minimum absolute scale to consider a model non-degenerate.
        # Linear: metres per rel-depth unit.  Inverse: metres * rel-depth unit.
        MIN_SCALE = {
            'linear': 0.5,
            'inverse': 1.0,
        }

        def _median_linear_scale():
            """Return a robust linear scale and zero shift."""
            ratios = lidar_vals / (rel_vals + 1e-6)
            return float(np.median(ratios)), 0.0

        def _median_inverse_scale():
            """Return a robust inverse scale and zero shift."""
            inv_ratios = lidar_vals * (rel_vals + 1e-6)
            return float(np.median(inv_ratios)), 0.0

        if n_valid >= min_points:
            # Try linear model first: metric = scale * rel + shift
            A = np.stack([rel_vals, np.ones_like(rel_vals)], axis=1)
            scale, shift = np.linalg.lstsq(A, lidar_vals, rcond=None)[0]
            mode = 'linear'

            # MiDaS outputs inverse-like relative depth (larger rel = closer).
            # If scale is negative, switch to inverse model:
            # metric = scale / rel + shift
            if scale < 0:
                inv_rel_vals = 1.0 / (rel_vals + 1e-6)
                A_inv = np.stack([inv_rel_vals, np.ones_like(inv_rel_vals)], axis=1)
                scale, shift = np.linalg.lstsq(A_inv, lidar_vals, rcond=None)[0]
                mode = 'inverse'

            # A very large positive shift acts as a near-constant floor: even
            # distant pixels get metric >= shift, which inflates mid-range
            # occupancy. Reject such fits for BOTH model families (a small
            # relative-depth dynamic range makes the linear fit do this too).
            if mode in ('linear', 'inverse') and shift > 10.0:
                if mode == 'linear':
                    scale, shift = _median_linear_scale()
                    mode = 'linear_median'
                else:
                    scale, shift = _median_inverse_scale()
                    mode = 'inverse_median'

            # Degenerate fit: scale ~= 0 means metric depth is essentially a
            # constant shift, producing a phantom far wall. Fall back to a
            # robust median-ratio model of the same family.
            if abs(scale) < MIN_SCALE[mode.replace('_median', '')]:
                if mode.startswith('linear'):
                    scale, shift = _median_linear_scale()
                else:
                    scale, shift = _median_inverse_scale()
                mode = f'{mode.replace("_median", "")}_median'
        else:
            # Fallback: only scale, assume shift = 0. Try linear then inverse.
            scale, shift = _median_linear_scale()
            mode = 'linear'
            if scale < 0:
                scale, shift = _median_inverse_scale()
                mode = 'inverse'
            if abs(scale) < MIN_SCALE[mode]:
                mode = f'{mode}_median'

        if mode.startswith('linear'):
            depth_metric = scale * depth_rel + shift
        else:
            depth_metric = scale / (depth_rel + 1e-6) + shift

        depth_metric = np.clip(depth_metric, 0.0, max_metric_depth)
        print(f"[align_to_lidar] n_valid={n_valid}, {mode}: scale={scale:.3f}, shift={shift:.3f}")
        return depth_metric.astype(np.float32)

    def infer(
        self,
        image: np.ndarray,
        lidar_sparse_depth: Optional[np.ndarray] = None,
        target_size: Optional[tuple] = None,
    ) -> np.ndarray:
        """Predict a metric depth map.

        If ``lidar_sparse_depth`` is provided, align relative depth to metric
        scale. Otherwise return raw relative depth (not usable for ego
        back-projection).

        Parameters
        ----------
        image : np.ndarray
            (H, W, 3) uint8 RGB image.
        lidar_sparse_depth : np.ndarray, optional
            (H, W) sparse LiDAR depth for alignment.
        target_size : tuple, optional
            Optional ``(H, W)`` output size.

        Returns
        -------
        np.ndarray
            (H, W) float32 depth map.
        """
        depth_rel = self.infer_relative(image, target_size=target_size)

        if lidar_sparse_depth is not None:
            if lidar_sparse_depth.shape != depth_rel.shape:
                raise ValueError(
                    f"lidar_sparse_depth shape {lidar_sparse_depth.shape} "
                    f"does not match depth shape {depth_rel.shape}"
                )
            return self.align_to_lidar(depth_rel, lidar_sparse_depth)

        return depth_rel


class MockDepthEstimator:
    """Stand-in depth estimator used when no network/model is available.

    Returns the provided densified LiDAR depth map as the "predicted" metric
    depth. This lets the rest of the image-to-BEV pipeline be built and
    tested without waiting for a real monocular depth model.

    Examples
    --------
    >>> est = MockDepthEstimator()
    >>> depth_metric = est.infer(image, lidar_dense_depth=dense_depth)
    """

    def __init__(self):
        pass

    def infer(
        self,
        image: np.ndarray,
        lidar_dense_depth: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Return a metric depth map.

        Parameters
        ----------
        image : np.ndarray
            (H, W, 3) uint8 RGB image (used only for shape checking).
        lidar_dense_depth : np.ndarray, optional
            (H, W) float32 metric depth. If ``None``, a constant-depth
            placeholder is returned.

        Returns
        -------
        np.ndarray
            (H, W) float32 metric depth map.
        """
        H, W = image.shape[:2]
        if lidar_dense_depth is not None:
            if lidar_dense_depth.shape[:2] != (H, W):
                raise ValueError(
                    f"lidar_dense_depth shape {lidar_dense_depth.shape} "
                    f"does not match image shape {(H, W)}"
                )
            return lidar_dense_depth.astype(np.float32)

        # Last resort: constant 20 m placeholder so downstream code runs.
        print("[MockDepthEstimator] warning: no depth input, returning constant 20m.")
        return np.full((H, W), 20.0, dtype=np.float32)
