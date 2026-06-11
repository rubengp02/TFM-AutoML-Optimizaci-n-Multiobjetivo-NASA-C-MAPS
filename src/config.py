"""Constantes compartidas del proyecto."""

from __future__ import annotations

# Constantes generales.
RANDOM_STATE: int = 42 # Para reproducibilidad.
DEFAULT_VALIDATION_SIZE: float = 0.2
LOW_VARIANCE_THRESHOLD: float = 1e-8

# Objetivos para la optimización multi-objetivo.
DEFAULT_MULTI_OBJECTIVES: list[str] = ["RMSE", "train_time_s", "inference_time_s"]

# Parámetros por defecto para H2O AutoML.
DEFAULT_H2O_NFOLDS: int = 5
DEFAULT_H2O_MAX_MODELS: int = 10

# Parámetros por defecto para NSGA-II.
DEFAULT_NSGA2_TRIALS: int = 80
DEFAULT_NSGA2_POPULATION_SIZE: int = 16
DEFAULT_MAX_RMSE_DEGRADATION: float = 0.15

# Modelos soportados por NSGA-II.
SUPPORTED_NSGA2_MODELS: set[str] = {
    "LightGBM",
    "XGBoost",
    "HistGradientBoosting",
    "RandomForest",
    "ExtraTrees",
    "Ridge",
}

# Tokens para identificar modelos de ensamble, a los cuales no se les puede aplicar NSGA-II.
ENSEMBLE_TOKENS: tuple[str, ...] = ("ensemble", "stacked", "stacking")
