"""Debug: locate the exact attribute path of unpicklable objects in the
ResWorld train dataset (fix for 'cannot pickle dict_keys object').

Usage (from repo root, resworld env):
    python tools/debug_pickle.py
Prints lines like:
    dataset.eval_config.something [dict_keys]: cannot pickle 'dict_keys' object
"""
import os
import pickle
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Mimic tools/train.py: only the repo root is on sys.path, and the plugin is
# imported as projects.mmdet3d_plugin. Adding REPO/projects as well would make
# the plugin load under two module names and crash with
# "HungarianAssigner3D is already registered in bbox_assigner".
sys.path.insert(0, REPO)

import projects.mmdet3d_plugin  # noqa: E402  (registers custom datasets/models)
from mmcv import Config  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402

cfg = Config.fromfile(os.path.join(REPO, 'projects/configs/resworld/resworld_config.py'))
ds = build_dataset(cfg.data.train)


def scan(obj, path, depth, seen):
    if depth > 6 or id(obj) in seen:
        return
    seen.add(id(obj))
    try:
        pickle.dumps(obj)
        return
    except Exception:
        pass
    if hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.items():
            try:
                pickle.dumps(v)
                scan(v, '{}.{}'.format(path, k), depth + 1, seen)
            except Exception as e:
                print('{} .{} [{}]: {}'.format(path, k, type(v).__name__, e))
                scan(v, '{}.{}'.format(path, k), depth + 1, seen)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            try:
                pickle.dumps(v)
                scan(v, '{}[{}]'.format(path, k), depth + 1, seen)
            except Exception as e:
                print('{}[{}] [{}]: {}'.format(path, k, type(v).__name__, e))
                scan(v, '{}[{}]'.format(path, k), depth + 1, seen)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            try:
                pickle.dumps(v)
                scan(v, '{}[{}]'.format(path, i), depth + 1, seen)
            except Exception as e:
                print('{}[{}] [{}]: {}'.format(path, i, type(v).__name__, e))
                scan(v, '{}[{}]'.format(path, i), depth + 1, seen)


scan(ds, 'dataset', 0, set())
print('scan done')
