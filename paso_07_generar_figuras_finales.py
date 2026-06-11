"""PASO 07: generación de figuras finales para la memoria del TFM."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.experiment_tracking import clean_directory_contents_for_stable_run


RELEVANT_LABELS = {"LightGBM", "H2O Stacked Ensemble", "Ridge", "LightGBM NSGA-II T54", "LightGBM NSGA-II T27"}
PARETO_FRONT_LABELS = {"LightGBM", "Ridge", "LightGBM NSGA-II T54", "LightGBM NSGA-II T27"}
APPROACH_STYLES = {
    "Traditional baseline": {"color": "#1f77b4", "marker": "o"},
    "H2O AutoML": {"color": "#2ca02c", "marker": "s"},
    "NSGA-II": {"color": "#d62728", "marker": "^"},
}
STABLE_RUN_IDS = {"FD004_baseline", "FD004_h2o_30models", "FD004_comparison_initial", "FD004_compromise_selection", "FD004_nsga2_60trials", "FD004_comparison_final"}


# CARGA DE RESULTADOS FINALES

def load_final_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga las tablas finales de comparación y Pareto desde la carpeta de resultados.
    
    Argumentos:
        results_dir: Directorio base que contiene los archivos CSV finales.
    
    Devuelve:
        Tupla con (comparison_df, pareto_df).
    """
    comparison_path = results_dir / "comparison_latex.csv"
    pareto_path = results_dir / "pareto.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"No se encontró: {comparison_path}")
    if not pareto_path.exists():
        raise FileNotFoundError(f"No se encontró: {pareto_path}")
    comparison_df = pd.read_csv(comparison_path)
    pareto_df = pd.read_csv(pareto_path)
    if "is_pareto_optimal" not in comparison_df.columns:
        comparison_df["is_pareto_optimal"] = comparison_df["model_short"].isin(pareto_df["model_short"])
    return comparison_df, pareto_df


# UTILIDADES DE FIGURAS

def _to_bool_series(series: pd.Series) -> pd.Series:
    """Convierte una columna de Pareto a booleano de forma robusta."""
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "si", "sí"])


def _annotate_selected_points(ax, dataframe: pd.DataFrame, x_col: str, y_col: str, label_col: str, label_mask: pd.Series) -> None:
    """Anota filas seleccionadas con pequeños desplazamientos para reducir solapamientos."""
    offsets = [(6, 6), (8, -8), (-10, 8), (8, 12), (-9, -10)]
    labeled_rows_df = dataframe[label_mask].reset_index(drop=True)
    for row_index, (_, row) in enumerate(labeled_rows_df.iterrows()):
        x_offset, y_offset = offsets[row_index % len(offsets)]
        ax.annotate(str(row[label_col]), (row[x_col], row[y_col]), textcoords="offset points", xytext=(x_offset, y_offset), fontsize=12)


def _plot_metric_projection(dataframe: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, output_path: Path, log_x: bool = False, log_y: bool = False, label_names: set[str] | None = None) -> Path:
    """Crea una proyección 2D con enfoques diferenciados y Pareto resaltado."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = dataframe.copy()
    pareto_mask = _to_bool_series(plot_df["is_pareto_optimal"])

    fig, ax = plt.subplots(figsize=(10, 6.2))
    for approach_name, style in APPROACH_STYLES.items():
        approach_mask = plot_df["approach"].astype(str).eq(approach_name)
        if not approach_mask.any():
            continue
        ax.scatter(
            plot_df.loc[approach_mask, x_col],
            plot_df.loc[approach_mask, y_col],
            s=72,
            marker=style["marker"],
            color=style["color"],
            alpha=0.78,
            label=approach_name,
        )

    ax.scatter(
        plot_df.loc[pareto_mask, x_col],
        plot_df.loc[pareto_mask, y_col],
        s=118,
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
        label="Solución Pareto",
    )
    labels_to_show = label_names or RELEVANT_LABELS
    label_mask = plot_df["model_short"].isin(labels_to_show)
    _annotate_selected_points(ax, plot_df, x_col=x_col, y_col=y_col, label_col="model_short", label_mask=label_mask)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _copy_figure(source_path: Path, destination_path: Path) -> Path:
    """Copia una figura para mantener compatibilidad con nombres antiguos."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


# FIGURA 1

def plot_rmse_vs_train_time(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica RMSE frente al tiempo de entrenamiento para todos los modelos finales."""
    return _plot_metric_projection(
        dataframe=comparison_df,
        x_col="train_time_s",
        y_col="RMSE",
        title="RMSE frente al tiempo de entrenamiento",
        xlabel="Tiempo de entrenamiento (s, escala logarítmica)",
        ylabel="RMSE",
        output_path=output_dir / "rmse_vs_train_time_final.png",
        log_x=True,
    )


# FIGURA 2

def plot_rmse_vs_inference_time(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica RMSE frente al tiempo de inferencia para todos los modelos finales."""
    return _plot_metric_projection(
        dataframe=comparison_df,
        x_col="inference_time_s",
        y_col="RMSE",
        title="RMSE frente al tiempo de inferencia",
        xlabel="Tiempo de inferencia (s, escala logarítmica)",
        ylabel="RMSE",
        output_path=output_dir / "rmse_vs_inference_time_final.png",
        log_x=True,
    )


def plot_rmse_vs_inference_time_pareto(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica RMSE frente a inferencia solo con soluciones no dominadas."""
    pareto_df = comparison_df[_to_bool_series(comparison_df["is_pareto_optimal"])].copy().reset_index(drop=True)
    return _plot_metric_projection(
        dataframe=pareto_df,
        x_col="inference_time_s",
        y_col="RMSE",
        title="RMSE frente al tiempo de inferencia - soluciones Pareto",
        xlabel="Tiempo de inferencia (s, escala logarítmica)",
        ylabel="RMSE",
        output_path=output_dir / "rmse_vs_inference_time_pareto.png",
        log_x=True,
        label_names=PARETO_FRONT_LABELS,
    )


# FIGURA 3

def plot_pareto_front_full(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica RMSE frente a entrenamiento solo con soluciones no dominadas."""
    pareto_df = comparison_df[_to_bool_series(comparison_df["is_pareto_optimal"])].copy().reset_index(drop=True)
    return _plot_metric_projection(
        dataframe=pareto_df,
        x_col="train_time_s",
        y_col="RMSE",
        title="Frente de Pareto final completo",
        xlabel="Tiempo de entrenamiento (s, escala logarítmica)",
        ylabel="RMSE",
        output_path=output_dir / "pareto_front_full.png",
        log_x=True,
        label_names=PARETO_FRONT_LABELS,
    )


def plot_pareto_front_zoom(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica la región competitiva solo con soluciones no dominadas."""
    pareto_df = comparison_df[_to_bool_series(comparison_df["is_pareto_optimal"])].copy()
    zoom_df = pareto_df[pareto_df["RMSE"].astype(float) <= 40.5].copy().reset_index(drop=True)
    return _plot_metric_projection(
        dataframe=zoom_df,
        x_col="train_time_s",
        y_col="RMSE",
        title="Frente de Pareto final - región competitiva",
        xlabel="Tiempo de entrenamiento (s, escala logarítmica)",
        ylabel="RMSE",
        output_path=output_dir / "pareto_front_zoom.png",
        log_x=True,
        label_names=PARETO_FRONT_LABELS,
    )


def plot_train_time_vs_inference_time(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    """Grafica tiempo de entrenamiento frente a tiempo de inferencia para todos los modelos."""
    return _plot_metric_projection(
        dataframe=comparison_df,
        x_col="train_time_s",
        y_col="inference_time_s",
        title="Tiempo de entrenamiento frente a tiempo de inferencia",
        xlabel="Tiempo de entrenamiento (s, escala logarítmica)",
        ylabel="Tiempo de inferencia (s, escala logarítmica)",
        output_path=output_dir / "train_time_vs_inference_time_final.png",
        log_x=True,
        log_y=True,
    )


def generate_final_figures(results_dir: Path) -> list[Path]:
    """Genera todas las figuras finales del TFM desde los CSV finales de comparación y Pareto."""
    comparison_df, pareto_df = load_final_results(results_dir)
    output_dir = results_dir / "figures_final"
    comparison_df = comparison_df.copy()
    comparison_df["is_pareto_optimal"] = _to_bool_series(comparison_df["is_pareto_optimal"])
    pareto_df = pareto_df.copy()
    pareto_df["is_pareto_optimal"] = True

    pareto_front_full_path = plot_pareto_front_full(comparison_df, output_dir)
    train_vs_inference_path = plot_train_time_vs_inference_time(comparison_df, output_dir)
    old_pareto_path = _copy_figure(pareto_front_full_path, output_dir / "pareto_front_final.png")
    old_train_inference_path = _copy_figure(train_vs_inference_path, output_dir / "pareto_front_train_inference.png")

    return [
        pareto_front_full_path,
        plot_pareto_front_zoom(comparison_df, output_dir),
        plot_rmse_vs_train_time(comparison_df, output_dir),
        plot_rmse_vs_inference_time(comparison_df, output_dir),
        plot_rmse_vs_inference_time_pareto(comparison_df, output_dir),
        train_vs_inference_path,
        old_pareto_path,
        old_train_inference_path,
    ]


def run_paso_07_pipeline(results_dir: str = "results/final") -> list[Path]:
    """Genera figuras finales del TFM a partir de resultados consolidados."""
    figures_output_dir = Path(results_dir) / "figures_final"
    clean_directory_contents_for_stable_run(figures_output_dir, stable_run_ids=STABLE_RUN_IDS)
    output_paths = generate_final_figures(Path(results_dir))
    print("\n[PASO 07] Figuras generadas:")
    for output_path in output_paths:
        print(f"- {output_path}")
    return output_paths


def main() -> None:
    """Ejecuta la configuración final del paso 7: generación de figuras finales."""
    # Configuración usada en este experimento.
    output_dir = "results"
    final_comparison_run_id = "FD004_comparison_final"
    results_dir = Path(output_dir) / "runs" / final_comparison_run_id
    run_paso_07_pipeline(results_dir=results_dir)


if __name__ == "__main__":
    main()
