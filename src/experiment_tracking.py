"""Utilidades para gestionar run IDs, carpetas de ejecución y configuraciones."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def _sanitize_token(value: str) -> str:
    """Sanea texto para nombres de carpeta seguros."""
    return re.sub(r"[^A-Za-z0-9_-]+", "", value)


def generate_run_id(subset: str, execution_type: str, main_descriptor: str | None = None, timestamp: datetime | None = None) -> str:
    """Genera un run ID único y legible."""
    timestamp_text = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    subset_token = _sanitize_token(subset.upper())
    execution_type_token = _sanitize_token(execution_type.lower())
    parts = [subset_token, execution_type_token]
    if main_descriptor:
        parts.append(_sanitize_token(main_descriptor.lower()))
    parts.append(timestamp_text)
    return "_".join([p for p in parts if p])


def create_run_directory(results_root: Path, run_id: str) -> Path:
    """Crea y devuelve results/runs/{run_id}."""
    run_dir = results_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def clean_directory_contents_for_stable_run(target_dir: Path, stable_run_ids: set[str]) -> Path:
    resolved_target_dir = target_dir.resolve()
    parts = list(resolved_target_dir.parts)
    lowered_parts = [part.lower() for part in parts]
    if "runs" not in lowered_parts:
        raise ValueError(f"Ruta no válida para limpieza controlada: {resolved_target_dir}")
    runs_index = lowered_parts.index("runs")
    if runs_index == 0:
        raise ValueError(f"Ruta no válida para limpieza controlada: {resolved_target_dir}")
    if lowered_parts[runs_index - 1] != "results":
        raise ValueError(f"Ruta no válida para limpieza controlada: {resolved_target_dir}")
    if runs_index + 1 >= len(parts):
        raise ValueError(f"Ruta no válida para limpieza controlada: {resolved_target_dir}")
    run_id = parts[runs_index + 1]
    if run_id not in stable_run_ids:
        raise ValueError(f"run_id no permitido para limpieza controlada: {run_id}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in target_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return target_dir


def save_run_config(run_dir: Path, config: dict[str, Any]) -> Path:
    """Guarda la configuración de ejecución como JSON en la carpeta de la ejecución."""
    config_path = run_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config_path


def update_latest_files(results_root: Path, run_type: str, source_paths: list[Path]) -> Path:
    """Copia archivos generados a results/latest/{run_type}/."""
    latest_dir = results_root / "latest" / _sanitize_token(run_type.lower())
    latest_dir.mkdir(parents=True, exist_ok=True)

    for source_path in source_paths:
        if not source_path.exists():
            continue
        destination_path = latest_dir / source_path.name
        shutil.copy2(source_path, destination_path)

    return latest_dir


def update_latest_comparison_figures(results_root: Path, figures_dir: Path) -> Path:
    """Copia figuras de comparación a results/latest/comparison/figures/."""
    latest_figures_dir = results_root / "latest" / "comparison" / "figures"
    latest_figures_dir.mkdir(parents=True, exist_ok=True)

    if figures_dir.exists():
        for file in figures_dir.glob("*.png"):
            shutil.copy2(file, latest_figures_dir / file.name)

    return latest_figures_dir
