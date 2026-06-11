"""Funciones de exploración para el análisis de C-MAPSS."""

# IMPORTS

from __future__ import annotations

import numpy as np
import pandas as pd


# DIMENSIONES Y ESTRUCTURA DEL DATASET

def get_dimensions(train_df: pd.DataFrame, test_df: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Devuelve dimensiones de train, test e y_test."""
    return pd.DataFrame(
        [{"dataset": "train", "rows": train_df.shape[0], "cols": train_df.shape[1]},
        {"dataset": "test", "rows": test_df.shape[0], "cols": test_df.shape[1]},
        {"dataset": "y_test", "rows": y_test.shape[0], "cols": 1},
        ]
    )


def count_engines(df: pd.DataFrame) -> int:
    """Cuenta motores únicos (units) en un conjunto de datos."""
    return int(df["unit"].nunique())


def cycles_per_engine_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resume la distribución del número de ciclos por motor."""

    cycles_per_engine = df.groupby("unit")["cycle"].count().rename("cycle_count") # Agrupa por motor y cuenta ciclos tiene cada motor y se le da el nombre "cycle_count".
    return cycles_per_engine.describe().to_frame().T


# CALIDAD BÁSICA DE LOS DATOS

def null_values_summary(df: pd.DataFrame) -> pd.Series:
    """Devuelve el número de nulos por columna."""
    return df.isnull().sum()


def low_variance_columns(df: pd.DataFrame, threshold: float = 1e-8) -> pd.Series:
    """Detecta columnas numéricas con varianza menor o igual al umbral."""

    numeric_df = df.select_dtypes(include=[np.number])
    variances = numeric_df.var(numeric_only=True)
    return variances[variances <= threshold].sort_values()


# RESUMEN DE VARIABLES OPERATIVAS Y SENSORES

def operational_sensor_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve estadísticas descriptivas de ajustes operativos y sensores."""

    operational_columns = [column_name for column_name in df.columns if column_name.startswith("op_setting_")]
    sensor_columns = [column_name for column_name in df.columns if column_name.startswith("sensor_")]
    selected_columns = operational_columns + sensor_columns
    return df[selected_columns].describe().T


# EXPLORACIÓN COMPLETA

def run_basic_exploration(train_df: pd.DataFrame, test_df: pd.DataFrame, y_test: pd.Series, low_var_threshold: float = 1e-8) -> dict[str, pd.DataFrame | pd.Series | int]:
    """Devuelve un diccionario que contiene un bloque completo de exploración para experimentación de los modelos baseline."""

    return {
        "dimensions": get_dimensions(train_df, test_df, y_test),
        "n_engines_train": count_engines(train_df),
        "n_engines_test": count_engines(test_df),
        "cycles_train": cycles_per_engine_summary(train_df),
        "cycles_test": cycles_per_engine_summary(test_df),
        "nulls_train": null_values_summary(train_df),
        "nulls_test": null_values_summary(test_df),
        "low_var_train": low_variance_columns(train_df, threshold=low_var_threshold),
        "low_var_test": low_variance_columns(test_df, threshold=low_var_threshold),
        "stats_train": operational_sensor_summary(train_df),
        "stats_test": operational_sensor_summary(test_df),
    }
