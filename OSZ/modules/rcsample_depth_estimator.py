"""RCSample (BEVDepth4D-style) monocular depth estimator for OSZ.

Wraps the depth head of a trained ResWorld detector so OSZ masks can be
computed from the world model's *own* depth estimate instead of MiDaS. This
is the depth-source ablation arm: ``MiDaS (offline) vs RCSample (in-model)
vs LiDAR (upper bound)``, sharing the same OSZ geometry pipeline.

The estimator exposes the same interface as :class:`DepthEstimator`::

    infer(image, lidar_sparse_depth=None, target_size=None, K=None,
          T_cam2ego=None) -> (H, W) float32 metric depth

How it works
------------
1. Lazily build the ResWorld detector from ``config_path`` (mmcv Config) and
   load ``checkpoint_path`` (strict=False). Only ``img_backbone`` /
   ``img_neck`` / ``img_view_transformer.depth_net`` are used.
2. For one camera image: apply the *same* geometric pre-processing as the
   ResWorld pipeline (aspect-preserving resize to width 704, then vertical
   crop to 256 rows), run backbone+neck to get the 1/16 feature map, predict
   the D-class depth logits with the camera-aware DepthNet, softmax, and
   take the expected depth over the bin centres (metric, no scale alignment
   needed).
3. Map the depth back to the *original* image resolution (undo crop+resize
   via pad+interpolate) so it matches the original intrinsics used by the
   OSZ geometry.

Camera-aware conditioning
-------------------------
The DepthNet's 27-dim ``mlp_input`` must match the training distribution:
``post_rot``/``post_tran`` encode the crop+resize applied to the image
(resize scale ``s`` and crop offset ``[0, -crop_h]``), ``bda`` is identity
(inference convention), and ``intrin``/``sensor2ego`` are the real camera
intrinsics and camera->ego extrinsics.

Caveats
-------
- Requires the mmdet3d environment (mmcv + mmdet3d + the projects plugin);
  all mmdet3d imports are deferred to :meth:`_load` so the rest of the OSZ
  pipeline (pure numpy / MiDaS) does not depend on it.
- The depth range is bounded by the ResWorld ``grid_config['depth']``
  (1~35 m), so cells beyond it get clipped expectations.
- The RCSample depth is metric and trained in-domain, so
  ``lidar_sparse_depth`` is *not* used for alignment (kept for interface
  compatibility with the rest of the pipeline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - torch is required at runtime
    torch = None
    F = None
    PILImage = None

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Default ResWorld config (grid/depth/backbone definitions live there).
DEFAULT_CONFIG_PATH = str(
    _REPO_ROOT / "projects" / "configs" / "resworld" / "resworld_config.py"
)

#: ImageNet normalisation used by the ResWorld training pipeline (RGB order).
_IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class RCSampleDepthEstimator:
    """Metric depth from a trained ResWorld's RCSample depth head."""

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        checkpoint_path: str = "",
        device: Optional[str] = None,
        input_size: tuple = (256, 704),
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.input_size = input_size
        self.device = device or (
            "cuda" if self._cuda_available() else "cpu"
        )
        self._model = None  # built lazily in _load()
        self._backbone = None
        self._neck = None
        self._view_transformer = None
        self._depth_net = None
        self._bin_center = None

    @staticmethod
    def _cuda_available() -> bool:
        if torch is None:
            return False
        return torch.cuda.is_available()

    def _load(self):
        """Build the ResWorld detector and extract the depth sub-modules.

        Deferred imports: this module is importable without mmdet3d; the
        mmdet3d stack is only needed when the RCSample depth source is used.
        """
        if self._model is not None:
            return self

        if torch is None:
            raise RuntimeError("PyTorch is not available in this environment.")
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                "RCSample depth source needs a trained ResWorld checkpoint. "
                "Pass --rcsample-ckpt <path> (e.g. "
                "work_dirs/oa_resworld_config/epoch_12_ema.pth)."
            )
        if not Path(self.config_path).exists():
            raise FileNotFoundError(
                f"ResWorld config not found: {self.config_path}"
            )

        import mmcv  # noqa: F401  (mmcv required)
        from mmcv.runner import load_checkpoint
        from mmdet.models.builder import build_detector

        # Register ResWorld / RCSample / CustomFPN from the projects plugin.
        import projects.mmdet3d_plugin  # noqa: F401

        cfg = mmcv.Config.fromfile(self.config_path)
        model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
        load_checkpoint(
            model, self.checkpoint_path, map_location="cpu", strict=False
        )
        model.eval()
        model.to(self.device)

        self._model = model
        self._backbone = model.img_backbone
        self._neck = model.img_neck
        self._view_transformer = model.img_view_transformer
        self._depth_net = model.img_view_transformer.depth_net

        # Fail loudly if the config drifts from this estimator's hard-coded
        # geometric assumptions (they mirror the ResWorld training pipeline
        # and must stay in sync, see _make_mlp_input / _training_aug).
        vt_cfg = cfg.model["img_view_transformer"]
        assert tuple(vt_cfg["input_size"]) == tuple(self.input_size), (
            f"RCSampleDepthEstimator assumes input_size "
            f"{self.input_size}, but resworld_config sets "
            f"{tuple(vt_cfg['input_size'])} — resync."
        )
        dc = cfg.data_config
        assert dc.get("resize_test", 0.0) == 0.0, (
            f"resize_test must be 0 to match the offline estimator, "
            f"got {dc.get('resize_test')}."
        )
        assert tuple(dc.get("crop_h", (0.0, 0.0))) == (0.0, 0.0), (
            f"crop_h must be (0, 0) to match the offline estimator, "
            f"got {dc.get('crop_h')}."
        )

        # Expected-depth bin centres: depth = [d, d + interval).
        depth_cfg = cfg.model["img_view_transformer"]["grid_config"]["depth"]
        d = torch.arange(*depth_cfg, dtype=torch.float32)
        interval = float(depth_cfg[2])
        self._bin_center = (d + 0.5 * interval).to(self.device)

        print(
            f"[RCSampleDepthEstimator] loaded ResWorld from "
            f"{self.checkpoint_path} on {self.device}; "
            f"D={self._bin_center.numel()} bins over "
            f"[{depth_cfg[0]}, {depth_cfg[1]}) m"
        )
        return self

    @staticmethod
    def _make_mlp_input(K, T_cam2ego, post_rot, post_tran) -> np.ndarray:
        """27-dim camera-aware input exactly as RCSample.get_mlp_input.

        ``post_rot``/``post_tran`` encode the image crop+resize applied
        before the network (same convention as the training pipeline);
        ``bda`` is identity (inference-time convention).
        """
        intrin = np.asarray(K, dtype=np.float32)
        s2e = np.asarray(T_cam2ego, dtype=np.float32)
        post_rot = np.asarray(post_rot, dtype=np.float32)
        post_tran = np.asarray(post_tran, dtype=np.float32)
        bda = np.eye(3, dtype=np.float32)

        parts = [
            intrin[0, 0], intrin[1, 1], intrin[0, 2], intrin[1, 2],
            post_rot[0, 0], post_rot[0, 1], post_tran[0],
            post_rot[1, 0], post_rot[1, 1], post_tran[1],
            bda[0, 0], bda[0, 1], bda[1, 0], bda[1, 1], bda[2, 2],
        ]
        s2e_flat = s2e[:3, :].reshape(-1)  # 3x4 -> 12
        return np.concatenate([np.asarray(parts, dtype=np.float32), s2e_flat])

    @staticmethod
    def _training_aug(H: int, W: int, in_h: int, in_w: int):
        """Same geometric transform as the ResWorld test pipeline.

        Returns ``(s, newH, crop_h)``: aspect-preserving scale ``s``, the
        resized height ``newH`` and the vertical crop offset.
        """
        s = float(in_w) / float(W)          # resize = fW / W
        newH = int(round(H * s))
        crop_h = max(0, int(newH) - in_h)   # crop_h = newH - fH
        return s, newH, crop_h

    def _infer_core(
        self,
        image: np.ndarray,
        K: Optional[np.ndarray],
        T_cam2ego: Optional[np.ndarray],
        target_size: Optional[tuple],
        device: Optional[str] = None,
    ) -> "torch.Tensor":
        """Core prediction: returns a (H, W) metric depth tensor on device."""
        # Cheap input guard before building the (heavy) ResWorld model.
        H, W = image.shape[:2]
        out_h, out_w = target_size if target_size is not None else (H, W)
        in_h, in_w = self.input_size
        if W < in_w:
            raise ValueError(
                f"RCSampleDepthEstimator needs image width >= {in_w}, "
                f"got {W}. The training pipeline assumes 900x1600 sources."
            )

        self._load()
        if PILImage is None or F is None:
            raise RuntimeError("PIL / torch.nn.functional unavailable.")
        if K is None or T_cam2ego is None:
            print(
                "[RCSampleDepthEstimator] warning: K/T_cam2ego not provided, "
                "using identity for the camera-aware depth conditioning."
            )
            K = np.eye(3, dtype=np.float32)
            T_cam2ego = np.eye(4, dtype=np.float32)

        dev = device or self.device

        # Same geometric pre-processing as the ResWorld pipeline
        # (loading.py::img_transform_core): aspect-preserving resize with the
        # PIL default filter (BICUBIC — do NOT pass BILINEAR, it would deviate
        # from the training-time pixel sampling), then vertical crop to
        # input_size rows.
        s, newH, crop_h = self._training_aug(H, W, in_h, in_w)
        img_pil = PILImage.fromarray(image).resize((in_w, newH))
        img_pil = img_pil.crop((0, crop_h, in_w, crop_h + in_h))
        img = np.array(img_pil, dtype=np.float32)
        img = (img - _IMG_MEAN) / _IMG_STD
        tensor = (
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(dev)
        )  # (1, 3, 256, 704)

        # post_rot/post_tran encode the applied crop+resize (same convention
        # as loading.py::img_transform); depth_scale == 1 at test time.
        post_rot = np.eye(3, dtype=np.float32) * s
        post_rot[2, 2] = 1.0
        post_tran = np.array([0.0, -float(crop_h), 0.0], dtype=np.float32)
        mlp_input = torch.from_numpy(
            self._make_mlp_input(K, T_cam2ego, post_rot, post_tran)
        ).view(1, 1, -1).to(dev)

        with torch.no_grad():
            x = self._backbone(tensor)  # list of feature maps
            x = self._neck(x)
            if isinstance(x, (list, tuple)):
                x = x[0]
            # x: (1, C, 16, 44) at 1/16 resolution
            logits, _context, _depth_feat = self._depth_net(x, mlp_input)
            depth_prob = logits.softmax(dim=1)  # (1, D, 16, 44)
            metric = (depth_prob * self._bin_center.view(1, -1, 1, 1)).sum(
                dim=1, keepdim=True
            )  # (1, 1, 16, 44)
            metric = F.interpolate(
                metric, size=(in_h, in_w), mode="bilinear",
                align_corners=False,
            )
            # Undo the crop: the depth patch covers rows [crop_h, crop_h+in_h)
            # of the resized image. Pad the cropped-off top/bottom back (0 =
            # no depth there; OSZ ignores d<=0), then scale up to the original
            # resolution so it matches the original intrinsics.
            bottom = max(0, newH - (crop_h + in_h))
            metric = F.pad(metric, (0, 0, crop_h, bottom))  # (1,1,newH,in_w)
            metric = F.interpolate(
                metric, size=(H, W), mode="bilinear",
                align_corners=False,
            )[0, 0]
            # Zero out the cropped-off region: bilinear upsampling would
            # otherwise blend 0-pad into the crop boundary and back-project
            # phantom near-field points. Nearest keeps the mask binary.
            valid = torch.zeros((1, 1, newH, in_w), device=metric.device)
            valid[:, :, crop_h:crop_h + in_h, :] = 1.0
            valid = F.interpolate(
                valid, size=(H, W), mode="nearest")[0, 0]
            metric = metric * valid
            if target_size is not None:
                metric = F.interpolate(
                    metric[None, None], size=(out_h, out_w),
                    mode="bilinear", align_corners=False,
                )[0, 0]

        return metric

    def infer(
        self,
        image: np.ndarray,
        lidar_sparse_depth: Optional[np.ndarray] = None,
        target_size: Optional[tuple] = None,
        K: Optional[np.ndarray] = None,
        T_cam2ego: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Predict a metric depth map from one camera RGB image.

        Parameters
        ----------
        image : np.ndarray
            (H, W, 3) uint8 RGB image.
        lidar_sparse_depth : np.ndarray, optional
            Ignored (RCSample depth is already metric). Kept for interface
            compatibility.
        target_size : tuple, optional
            ``(H, W)`` to resize the output; defaults to the input size.
        K : np.ndarray, optional
            (3, 3) camera intrinsic; used for the camera-aware MLP input.
            Defaults to identity (slightly degrades the SE modulation only).
        T_cam2ego : np.ndarray, optional
            (4, 4) camera->ego extrinsic; same use. Defaults to identity.

        Returns
        -------
        np.ndarray
            (H, W) float32 metric depth map.
        """
        metric = self._infer_core(image, K, T_cam2ego, target_size)
        return metric.cpu().numpy().astype(np.float32)

    def infer_tensor(
        self,
        image: np.ndarray,
        lidar_sparse_depth: Optional[np.ndarray] = None,
        target_size: Optional[tuple] = None,
        K: Optional[np.ndarray] = None,
        T_cam2ego: Optional[np.ndarray] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> "torch.Tensor":
        """Like :meth:`infer` but returns the depth on the compute device."""
        return self._infer_core(image, K, T_cam2ego, target_size,
                                device=device)
