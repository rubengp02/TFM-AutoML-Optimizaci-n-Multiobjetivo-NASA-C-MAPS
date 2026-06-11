"""Utilidades de generación de figuras finales para el capítulo de resultados del TFM."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RELEVANT_LABELS = {
    "LightGBM NSGA-II T47",
    "H2O Stacked Ensemble",
    "LightGBM",
    "Ridge",
}


def load_final_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga las tablas finales de comparación y Pareto desde results/final.

    Argumentos:
        results_dir: Directorio base que contiene los archivos CSV finales.

    Devuelve:
        Tupla con (comparison_df, pareto_df).
    """
    comparison_path = results_dir / "comparison_FD004_final_latex.csv"
    pareto_path = results_dir / "pareto_FD004_final.csv"

    if not comparison_path.exists():
        raise FileNotFoundError(f"Final comparison file not found: {comparison_path}")
    if not pareto_path.exists():
        raise FileNotFoundError(f"Final Pareto file not found: {pareto_path}")

    comparison_df = pd.read_csv(comparison_path)
    pareto_df = pd.read_csv(pareto_path)

    if "is_pareto_optimal" not in comparison_df.columns:
        comparison_df["is_pareto_optimal"] = comparison_df["model_short"].isin(pareto_df["model_short"])

    return comparison_df, pareto_df


def _scale_point_sizes(values: pd.Series, min_size: float = 60.0, max_size: float = 320.0) -> np.ndarray:
    """Escala valores numéricos a tamaños de marcador para dispersión."""
    numeric_values = values.astype(float).to_numpy()
    minimum_value = float(np.min(numeric_values))
    maximum_value = float(np.max(numeric_values))
    if abs(maximum_value - minimum_value) <= 1e-12:
        return np.full_like(numeric_values, fill_value=(min_size + max_size) / 2.0, dtype=float)
    normalized_values = (numeric_values - minimum_value) / (maximum_value - minimum_value)
    return min_size + normalized_values * (max_size - min_size)


def _annotate_selected_points(ax, df: pd.DataFrame, x_col: str, y_col: str, label_col: str, label_mask: pd.Series) -> None:
    """Anota filas seleccionadas con pequeños desplazamientos para reducir solapamientos."""
    offsets = [(6, 6), (8, -8), (-10, 8), (8, 12), (-9, -10), (12, 0)]
    labeled_rows_df = df[label_mask].reset_index(drop=True)
    for row_index, (_, row) in enumerate(labeled_rows_df.iterrows()):
        x_offset, y_offset = offsets[row_index % len(offsets)]
        ax.annotate(
            str(row[label_col]),
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(x_offset, y_offset),
            fontsize=9,
        )


def plot_rmse_vs_train_time(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica RMSE frente al tiempo de entrenamiento para todos los modelos finales."""
    output_path = output_dir / "rmse_vs_train_time_final.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6.2))

    pareto_mask = comparison_df["is_pareto_optimal"].astype(bool)
    ax.scatter(comparison_df["train_time_s"], comparison_df["RMSE"], alpha=0.7, label="Modelos")
    ax.scatter(
        comparison_df.loc[pareto_mask, "train_time_s"],
        comparison_df.loc[pareto_mask, "RMSE"],
        marker="D",
        s=80,
        alpha=0.95,
        label="Pareto",
    )

    label_mask = pareto_mask | comparison_df["model_short"].isin(RELEVANT_LABELS)
    _annotate_selected_points(
        ax,
        comparison_df,
        x_col="train_time_s",
        y_col="RMSE",
        label_col="model_short",
        label_mask=label_mask,
    )

    ax.set_xscale("log")
    ax.set_title("RMSE frente al tiempo de entrenamiento")
    ax.set_xlabel("Tiempo de entrenamiento (s, escala logarítmica)")
    ax.set_ylabel("RMSE")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_rmse_vs_inference_time(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica RMSE frente al tiempo de inferencia para todos los modelos finales."""
    output_path = output_dir / "rmse_vs_inference_time_final.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6.2))

    pareto_mask = comparison_df["is_pareto_optimal"].astype(bool)
    ax.scatter(comparison_df["inference_time_s"], comparison_df["RMSE"], alpha=0.7, label="Modelos")
    ax.scatter(
        comparison_df.loc[pareto_mask, "inference_time_s"],
        comparison_df.loc[pareto_mask, "RMSE"],
        marker="D",
        s=80,
        alpha=0.95,
        label="Pareto",
    )

    label_mask = pareto_mask | comparison_df["model_short"].isin(RELEVANT_LABELS)
    _annotate_selected_points(
        ax,
        comparison_df,
        x_col="inference_time_s",
        y_col="RMSE",
        label_col="model_short",
        label_mask=label_mask,
    )

    ax.set_xscale("log")
    ax.set_title("RMSE frente al tiempo de inferencia")
    ax.set_xlabel("Tiempo de inferencia (s, escala logarítmica)")
    ax.set_ylabel("RMSE")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_pareto_front(pareto_df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica el frente de Pareto final con tamaño de marcador según tiempo de inferencia."""
    output_path = output_dir / "pareto_front_final.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    sizes = _scale_point_sizes(pareto_df["inference_time_s"])

    ax.scatter(pareto_df["train_time_s"], pareto_df["RMSE"], s=sizes, alpha=0.8)
    _annotate_selected_points(
        ax,
        pareto_df,
        x_col="train_time_s",
        y_col="RMSE",
        label_col="model_short",
        label_mask=pd.Series([True] * len(pareto_df)),
    )

    ax.set_xscale("log")
    ax.set_title("Frente de Pareto final")
    ax.set_xlabel("Tiempo de entrenamiento (s, escala logarítmica)")
    ax.set_ylabel("RMSE")
    ax.text(
        0.02,
        0.02,
        "Nota: el tamaño del punto representa el tiempo de inferencia.",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_pareto_train_vs_inference(pareto_df: pd.DataFrame, output_dir: Path) -> Path:
    """Gráfico opcional: entrenamiento vs inferencia en el frente de Pareto (tamaño=RMSE)."""
    output_path = output_dir / "pareto_front_train_inference.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    sizes = _scale_point_sizes(pareto_df["RMSE"])

    ax.scatter(pareto_df["train_time_s"], pareto_df["inference_time_s"], s=sizes, alpha=0.8)
    _annotate_selected_points(
        ax,
        pareto_df,
        x_col="train_time_s",
        y_col="inference_time_s",
        label_col="model_short",
        label_mask=pd.Series([True] * len(pareto_df)),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Compromiso entre entrenamiento e inferencia en el frente de Pareto")
    ax.set_xlabel("Tiempo de entrenamiento (s, escala logarítmica)")
    ax.set_ylabel("Tiempo de inferencia (s, escala logarítmica)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def generate_final_figures(results_dir: Path) -> list[Path]:
    """Genera todas las figuras finales del TFM desde los CSV finales de comparación y Pareto."""
    comparison_df, pareto_df = load_final_results(results_dir)
    output_dir = results_dir / "figures_final"

    outputs = [
        plot_rmse_vs_train_time(comparison_df, output_dir),
        plot_rmse_vs_inference_time(comparison_df, output_dir),
        plot_pareto_front(pareto_df, output_dir),
        plot_pareto_train_vs_inference(pareto_df, output_dir),
    ]
    return outputs
