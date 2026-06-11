"""Utilidades para combinar y analizar tablas de resultados experimentales."""

# IMPORTS
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# NORMALIZACIÓN DE NOMBRES DE MODELOS

def shorten_model_name(model_name: str) -> str:
    """Convierte el nombre largo dado por H2O en nombres más cortos para visualización."""

    name = str(model_name)
    if name.startswith("StackedEnsemble_AllModels"): # Nombre por defecto para el modelo de ensamblado de H2O.
        return "H2O Stacked Ensemble"

    mapping = {
        "LGBMRegressor": "LightGBM",
        "XGBRegressor": "XGBoost",
        "HistGradientBoostingRegressor": "HistGradientBoosting",
        "RandomForestRegressor": "RandomForest",
        "ExtraTreesRegressor": "ExtraTrees",
        "RidgePipeline": "Ridge",
    }

    return mapping.get(name, name)


# DETECCIÓN DEL FRENTE DE PARETO

def detect_pareto_solutions(df: pd.DataFrame, objectives: list[str]) -> pd.DataFrame:
    """Marca filas no dominadas (Pareto-óptimas) para objetivos de minimización."""
    data = df.copy().reset_index(drop=True)
    values = data[objectives].to_numpy(dtype=float)
    n = len(data)

    is_pareto = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            no_worse_all = (values[j] <= values[i]).all()
            strictly_better_one = (values[j] < values[i]).any()

            if no_worse_all and strictly_better_one:
                is_pareto[i] = False
                break

    data["is_pareto_optimal"] = is_pareto
    return data


# TABLAS PARA RESULTADOS Y LATEX

def build_compact_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Crea una vista compacta de comparación para informes."""
    selected_columns = [
        "model_short",
        "approach",
        "MAE",
        "RMSE",
        "R2",
        "train_time_s",
        "inference_time_s",
        "inference_time_per_sample_s",
        "n_features",
        "is_pareto_optimal",
    ]
    return df[selected_columns].copy().sort_values("RMSE", ascending=True).reset_index(drop=True)


def build_latex_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabla redondeada para exportación LaTeX."""
    latex_df = build_compact_comparison_table(df)
    latex_df["MAE"] = latex_df["MAE"].round(2)
    latex_df["RMSE"] = latex_df["RMSE"].round(2)
    latex_df["R2"] = latex_df["R2"].round(3)
    latex_df["train_time_s"] = latex_df["train_time_s"].round(3)
    latex_df["inference_time_s"] = latex_df["inference_time_s"].round(6)
    return latex_df


# FUNCIONES AUXILIARES PARA FIGURAS

def _maybe_set_log_x(ax, x_values: pd.Series) -> None:
    """Configura escala logarítmica en X cuando los tiempos varían mucho."""
    positive_values = x_values[x_values > 0]
    if positive_values.empty:
        return
    range_ratio = positive_values.max() / positive_values.min()
    if range_ratio >= 50:
        ax.set_xscale("log")


def _compute_label_offsets(df: pd.DataFrame, x_col: str, y_col: str) -> list[tuple[int, int]]:
    """Calcula desplazamientos por punto para reducir solapamiento de etiquetas en puntos cercanos."""
    x_range = max(float(df[x_col].max() - df[x_col].min()), 1e-12)
    y_range = max(float(df[y_col].max() - df[y_col].min()), 1e-12)
    x_tol = 0.08 * x_range
    y_tol = 0.08 * y_range

    base_offsets = [(6, 6), (8, -8), (-10, 7), (7, 11), (-9, -10), (11, 0), (0, 11), (-11, 0)]
    offsets: list[tuple[int, int]] = []

    for i, row_i in df.reset_index(drop=True).iterrows():
        close_count = 0
        for j in range(i):
            row_j = df.iloc[j]
            if abs(float(row_i[x_col]) - float(row_j[x_col])) <= x_tol and abs(float(row_i[y_col]) - float(row_j[y_col])) <= y_tol:
                close_count += 1

        dx, dy = base_offsets[close_count % len(base_offsets)]
        offsets.append((dx, dy))

    return offsets


# GENERACIÓN DE FIGURAS DE COMPARACIÓN

def create_scatter_figure(df: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: Path, label_only_pareto: bool = False) -> None:
    """Crea y guarda una figura de dispersión con mejor legibilidad de etiquetas."""
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.scatter(df[x_col], df[y_col])
    _maybe_set_log_x(ax, df[x_col])

    if label_only_pareto and "is_pareto_optimal" in df.columns:
        label_df = df[df["is_pareto_optimal"]].copy().reset_index(drop=True)
    else:
        label_df = df.reset_index(drop=True)

    offsets = _compute_label_offsets(label_df, x_col=x_col, y_col=y_col) if not label_df.empty else []
    for idx, (_, row) in enumerate(label_df.iterrows()):
        dx, dy = offsets[idx]
        ax.annotate(
            str(row["model_short"]),
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
