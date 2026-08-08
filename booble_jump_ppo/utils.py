import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import MODELS_ROOT


def choose_weighted(rng: random.Random, mapping: Dict[str, float]) -> str:
    t = rng.random()
    acc = 0.0
    for k, w in mapping.items():
        acc += w
        if t <= acc:
            return k
    return list(mapping.keys())[-1]


def _to_scalar(x, fallback=0.0, prefer='mean') -> float:
    try:
        arr = np.asarray(x)
        if arr.ndim == 0:
            return float(arr.item())
        if arr.size == 0:
            return float(fallback)
        if prefer == 'max':
            return float(arr.max())
        if prefer == 'sum':
            return float(arr.sum())
        return float(arr.mean())
    except Exception:
        try:
            return float(x)
        except Exception:
            return float(fallback)


def list_model_runs(models_root: Path = MODELS_ROOT) -> List[Tuple[int, Path]]:
    if not models_root.exists():
        return []
    runs: List[Tuple[int, Path]] = []
    for p in models_root.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"model_(\d+)", p.name)
        if not m:
            continue
        runs.append((int(m.group(1)), p))
    runs.sort(key=lambda x: x[0])
    return runs


def create_next_model_run_dir(models_root: Path = MODELS_ROOT) -> Tuple[int, Path]:
    models_root.mkdir(parents=True, exist_ok=True)
    runs = list_model_runs(models_root)
    next_n = (runs[-1][0] + 1) if runs else 1
    run_dir = models_root / f"model_{next_n}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return next_n, run_dir


def get_latest_model_run_dir(models_root: Path = MODELS_ROOT) -> Tuple[int, Path]:
    runs = list_model_runs(models_root)
    if not runs:
        raise FileNotFoundError(f"Нет папок вида {models_root}/model_N")
    return runs[-1]


def _extract_last_int(s: str) -> int:
    m = re.findall(r"\d+", s)
    return int(m[-1]) if m else -1


def resolve_model_path_in_run(run_dir: Path) -> Path:
    final_model = run_dir / "vec_final_model.pth"
    if final_model.exists():
        return final_model

    best = list(run_dir.glob("vec_best_score_*.pth"))
    if best:
        best.sort(key=lambda p: _extract_last_int(p.name))
        return best[-1]

    ckpt = list(run_dir.glob("vec_checkpoint_step_*.pth"))
    if ckpt:
        ckpt.sort(key=lambda p: _extract_last_int(p.name))
        return ckpt[-1]

    raise FileNotFoundError(f"В {run_dir} не найдено ни одной модели (.pth)")
