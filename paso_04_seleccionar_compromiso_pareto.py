"""PASO 04: selección automática de solución de compromiso Pareto."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import DEFAULT_MAX_RMSE_DEGRADATION, ENSEMBLE_TOKENS, SUPPORTED_NSGA2_MODELS
from src.cmapss_data_preparation import get_project_root
from src.experiment_tracking import clean_directory_contents_for_stable_run, create_run_directory, generate_run_id, save_run_config, update_latest_files

STABLE_RUN_IDS = {"FD004_baseline", "FD004_h2o_30models", "FD004_comparison_initial", "FD004_compromise_selection", "FD004_nsga2_60trials", "FD004_comparison_final"}


# FILTRADO Y SCORING DE CANDIDATOS PARETO

def _safe_minmax(series: pd.Series) -> pd.Series:
    """Aplica normalización min-max con tratamiento seguro de valores constantes."""
    minimum_value = float(series.min())
    maximum_value = float(series.max())
    if maximum_value - minimum_value <= 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - minimum_value) / (maximum_value - minimum_value)


def filter_supported_non_ensemble_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva filas Pareto-óptimas compatibles y que no sean ensemble/stacked."""
    if "is_pareto_optimal" not in df.columns:
        raise ValueError("Input dataframe must contain 'is_pareto_optimal'.")
    model_column = "model_short" if "model_short" in df.columns else "model"
    pareto_candidates_df = df[df["is_pareto_optimal"] == True].copy()  # noqa: E712
    # Antes de NSGA-II se excluyen ensembles para mantener modelos base ajustables.
    model_text = pareto_candidates_df[model_column].astype(str).str.lower()
    is_ensemble = model_text.apply(lambda name: any(token in name for token in ENSEMBLE_TOKENS))
    is_supported = pareto_candidates_df[model_column].astype(str).isin(SUPPORTED_NSGA2_MODELS)
    return pareto_candidates_df[~is_ensemble & is_supported].copy().reset_index(drop=True)


def select_best_compromise_solution(df: pd.DataFrame, objective_cols: Iterable[str], time_cols: Iterable[str], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Selecciona el mejor compromiso por distancia normalizada al punto ideal.
    
    Pasos:
    - Usa las columnas objetivo proporcionadas.
    - Aplica log1p a objetivos de tiempo antes de normalizar.
    - Aplica normalización min-max.
    - Calcula distancia euclídea ponderada al punto ideal (todo ceros).
    """
    objectives = list(objective_cols)
    time_set = set(time_cols)
    missing_columns = [objective_name for objective_name in objectives if objective_name not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing objective columns: {missing_columns}")

    scored_df = df.copy().reset_index(drop=True)
    # Build normalized columns required by the user.
    norm_column_map = {"RMSE": "RMSE_norm", "train_time_s": "train_time_s_norm", "inference_time_s": "inference_time_s_norm"}
    for objective_name in objectives:
        # log1p en tiempos para reducir sesgo por escalas temporales extremas.
        transformed = np.log1p(scored_df[objective_name].astype(float)) if objective_name in time_set else scored_df[objective_name].astype(float)
        scored_df[norm_column_map.get(objective_name, f"{objective_name}_norm")] = _safe_minmax(transformed)

    # Default equal weights on normalized objectives.
    objective_weights = {objective_name: float(weights.get(objective_name, 1.0)) for objective_name in objectives} if weights else {objective_name: 1.0 for objective_name in objectives}
    distance_squared = np.zeros(len(scored_df), dtype=float)
    for objective_name in objectives:
        normalized_column = norm_column_map.get(objective_name, f"{objective_name}_norm")
        distance_squared += objective_weights[objective_name] * np.square(scored_df[normalized_column].astype(float).to_numpy())

    # Weighted Euclidean distance to ideal point (0,0,...).
    scored_df["ideal_distance"] = np.sqrt(distance_squared)
    best_index = int(scored_df["ideal_distance"].idxmin())
    scored_df["is_best_compromise"] = False
    scored_df.loc[best_index, "is_best_compromise"] = True
    return scored_df


def choose_model_for_nsga2(comparison_df: pd.DataFrame, max_rmse_degradation: float = DEFAULT_MAX_RMSE_DEGRADATION) -> pd.DataFrame:
    """Selecciona candidatos para NSGA-II con un umbral de calidad por RMSE.
    
    Pasos:
    - Filtra modelos Pareto-óptimos, no-ensemble y compatibles.
    - Conserva solo filas dentro del umbral de degradación RMSE respecto al mejor RMSE.
    - Calcula distancia de compromiso sobre las filas filtradas.
    """
    candidates_df = filter_supported_non_ensemble_pareto(comparison_df)
    if candidates_df.empty:
        raise ValueError("No supported non-ensemble Pareto solutions available for compromise selection.")

    best_rmse = float(candidates_df["RMSE"].min())
    rmse_threshold = best_rmse * (1.0 + float(max_rmse_degradation))
    candidates_df = candidates_df.copy()
    candidates_df["passes_rmse_filter"] = candidates_df["RMSE"] <= rmse_threshold
    filtered_candidates_df = candidates_df[candidates_df["passes_rmse_filter"]].copy().reset_index(drop=True)
    if filtered_candidates_df.empty:
        raise ValueError(f"No candidates passed RMSE degradation filter. Threshold={rmse_threshold:.6f}")

    selected_df = select_best_compromise_solution(
        df=filtered_candidates_df,
        objective_cols=["RMSE", "train_time_s", "inference_time_s"],
        time_cols=["train_time_s", "inference_time_s"],
        weights=None,
    )
    return selected_df.sort_values("ideal_distance", ascending=True).reset_index(drop=True)


def get_best_compromise_row(comparison_df: pd.DataFrame, max_rmse_degradation: float = DEFAULT_MAX_RMSE_DEGRADATION) -> tuple[pd.Series, pd.DataFrame]:
    """Devuelve la mejor fila de compromiso y la tabla ordenada de candidatos para configurar NSGA-II."""
    candidates_df = choose_model_for_nsga2(comparison_df=comparison_df, max_rmse_degradation=max_rmse_degradation)
    if candidates_df.empty:
        raise ValueError("No compromise candidates available after filtering and scoring.")
    return candidates_df.iloc[0], candidates_df


# ORQUESTACIÓN PASO 04

def run_paso_04_pipeline(subset: str, comparison_run_id: str, max_rmse_degradation: float = DEFAULT_MAX_RMSE_DEGRADATION, run_id: str | None = None, output_dir: str = "results") -> dict[str, str]:
    """Ejecuta la selección automática de compromiso y guarda resultados."""
    subset = subset.upper()
    project_root = get_project_root()
    results_root = Path(project_root) / output_dir
    comparison_path = results_root / "runs" / comparison_run_id / "comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"No se encontró comparison.csv: {comparison_path}")

    run_id = run_id or generate_run_id(subset=subset, execution_type="compromise")
    run_dir = create_run_directory(results_root=results_root, run_id=run_id)
    clean_directory_contents_for_stable_run(run_dir, stable_run_ids=STABLE_RUN_IDS)

    comparison_df = pd.read_csv(comparison_path)
    best_row, candidates_df = get_best_compromise_row(
        comparison_df=comparison_df,
        max_rmse_degradation=max_rmse_degradation,
    )
    output_path = run_dir / "compromise_selection.csv"
    candidates_df.to_csv(output_path, index=False)

    config_path = save_run_config(
        run_dir,
        {
            "run_id": run_id,
            "subset": subset,
            "execution_type": "compromise_selection",
            "comparison_run_id": comparison_run_id,
            "max_rmse_degradation": max_rmse_degradation,
            "generated_files": [
                "compromise_selection.csv",
                "run_config.json",
            ],
        },
    )
    latest_dir = update_latest_files(
        results_root=results_root,
        run_type="compromise",
        source_paths=[output_path, config_path],
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "output_path": str(output_path),
        "latest_dir": str(latest_dir),
        "selected_model_short": str(best_row.get("model_short", "")),
    }


def main() -> None:
    """Parsea argumentos CLI y selecciona la mejor solución de compromiso del Pareto inicial."""
    # Configuración usada en este experimento.
    subset = "FD004"
    comparison_run_id = "FD004_comparison_initial"
    max_rmse_degradation = DEFAULT_MAX_RMSE_DEGRADATION
    run_id = "FD004_compromise_selection"
    output_dir = "results"
    outputs = run_paso_04_pipeline(
        subset=subset,
        comparison_run_id=comparison_run_id,
        max_rmse_degradation=max_rmse_degradation,
        run_id=run_id,
        output_dir=output_dir,
    )
    print("\n[PASO 04] Ejecutada correctamente")
    print(f"[PASO 04] run_id: {outputs['run_id']}")
    print(f"[PASO 04] run_dir: {outputs['run_dir']}")
    print(f"[PASO 04] model_short seleccionado: {outputs['selected_model_short']}")


if __name__ == "__main__":
    main()
