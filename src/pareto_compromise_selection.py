"""Utilidades para seleccionar una solución de compromiso automática entre candidatos de Pareto."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.config import DEFAULT_MAX_RMSE_DEGRADATION, ENSEMBLE_TOKENS, SUPPORTED_NSGA2_MODELS


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

    model_col = "model_short" if "model_short" in df.columns else "model"

    pareto = df[df["is_pareto_optimal"] == True].copy()  # noqa: E712

    model_text = pareto[model_col].astype(str).str.lower()
    is_ensemble = model_text.apply(lambda name: any(token in name for token in ENSEMBLE_TOKENS))

    supported = pareto[model_col].astype(str).isin(SUPPORTED_NSGA2_MODELS)

    return pareto[~is_ensemble & supported].copy().reset_index(drop=True)


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
    norm_col_map = {
        "RMSE": "RMSE_norm",
        "train_time_s": "train_time_s_norm",
        "inference_time_s": "inference_time_s_norm",
    }

    normalized_cols: list[str] = []
    for objective_name in objectives:
        transformed = np.log1p(scored_df[objective_name].astype(float)) if objective_name in time_set else scored_df[objective_name].astype(float)
        norm_values = _safe_minmax(transformed)
        target_norm_col = norm_col_map.get(objective_name, f"{objective_name}_norm")
        scored_df[target_norm_col] = norm_values
        normalized_cols.append(target_norm_col)

    # Default equal weights on normalized objectives.
    if weights is None:
        used_weights = {objective_name: 1.0 for objective_name in objectives}
    else:
        used_weights = {objective_name: float(weights.get(objective_name, 1.0)) for objective_name in objectives}

    # Weighted Euclidean distance to ideal point (0,0,...).
    distance_squared = np.zeros(len(scored_df), dtype=float)
    for objective_name in objectives:
        normalized_column = norm_col_map.get(objective_name, f"{objective_name}_norm")
        objective_weight = used_weights[objective_name]
        distance_squared += objective_weight * np.square(scored_df[normalized_column].astype(float).to_numpy())

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
    candidates = filter_supported_non_ensemble_pareto(comparison_df)
    if candidates.empty:
        raise ValueError("No supported non-ensemble Pareto solutions available for compromise selection.")

    best_rmse = float(candidates["RMSE"].min())
    rmse_threshold = best_rmse * (1.0 + float(max_rmse_degradation))

    candidates = candidates.copy()
    candidates["passes_rmse_filter"] = candidates["RMSE"] <= rmse_threshold
    filtered = candidates[candidates["passes_rmse_filter"]].copy().reset_index(drop=True)

    if filtered.empty:
        raise ValueError(
            "No candidates passed the RMSE degradation filter. "
            f"Threshold={rmse_threshold:.6f}, best_rmse={best_rmse:.6f}, "
            f"max_rmse_degradation={max_rmse_degradation}."
        )

    selected = select_best_compromise_solution(
        df=filtered,
        objective_cols=["RMSE", "train_time_s", "inference_time_s"],
        time_cols=["train_time_s", "inference_time_s"],
        weights=None,
    )

    return selected.sort_values("ideal_distance", ascending=True).reset_index(drop=True)


def get_best_compromise_row(comparison_df: pd.DataFrame, max_rmse_degradation: float = DEFAULT_MAX_RMSE_DEGRADATION) -> tuple[pd.Series, pd.DataFrame]:
    """Devuelve la mejor fila de compromiso y la tabla ordenada de candidatos para configurar NSGA-II."""
    candidates_df = choose_model_for_nsga2(
        comparison_df=comparison_df,
        max_rmse_degradation=max_rmse_degradation,
    )
    if candidates_df.empty:
        raise ValueError("No compromise candidates available after filtering and scoring.")

    best_row = candidates_df.iloc[0]
    return best_row, candidates_df
