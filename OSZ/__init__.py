"""
OSZ/
====
Occlusion Shadow Zone (OSZ) pipeline for BEV height-aware occlusion computation.

Sub‑packages
  modules/    : Core computation steps (depth, projection, BEV, ray‑casting, filter)
  utils/      : Geometry helpers, NuScenes data loader
  visualize/  : BEV visualisation and sample inspection tools

Top‑level modules
  config.py              : Pipeline‑wide configuration (paths, thresholds, flags)
  run_osz_pipeline.py    : Entry point that runs the full OSZ pipeline end‑to‑end
"""