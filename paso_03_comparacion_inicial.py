"""PASO 03: comparación inicial (tradicionales + H2O) y Pareto inicial."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.cmapss_data_preparation import get_project_root
from src.experiment_tracking import clean_directory_contents_for_stable_run, create_run_directory, generate_run_id, save_run_config, update_latest_comparison_figures, update_latest_files

STABLE_RUN_IDS = {"FD004_baseline", "FD004_h2o_30models", "FD004_comparison_initial", "FD004_compromise_selection", "FD004_nsga2_60trials", "FD004_comparison_final"}


# FUNCIONES DE COMPARACIÓN

def shorten_model_name(model_name: str) -> str:
    """Convierte el nombre largo dado por H2O en nombres más cortos para visualización."""
    name = str(model_name)
    # Los modelos StackedEnsemble_AllModels de H2O se renombran para que tablas y figuras sean legibles.
    if name.startswith("StackedEnsemble_AllModels"):
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


def detect_pareto_solutions(df: pd.DataFrame, objectives: list[str]) -> pd.DataFrame:
    """Marca filas no dominadas (Pareto-óptimas) para objetivos de minimización."""
    # Dominancia de Pareto para objetivos de minimización.
    data = df.copy().reset_index(drop=True)
    values = data[objectives].to_numpy(dtype=float)
    is_pareto = [True] * len(data)
    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                continue
            no_worse_all = (values[j] <= values[i]).all()
            strictly_better_one = (values[j] < values[i]).any()
            if no_worse_all and strictly_better_one:
                is_pareto[i] = False
                break
    data["is_pareto_optimal"] = is_pareto
    return data


def build_compact_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Crea una vista compacta de comparación para informes."""
    selected_columns = ["model_short", "approach", "MAE", "RMSE", "R2", "train_time_s", "search_time_s", "inference_time_s", "inference_time_per_sample_s", "n_features", "is_pareto_optimal"]
    return df[selected_columns].copy().sort_values("RMSE", ascending=True).reset_index(drop=True)


def build_latex_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabla redondeada para exportación LaTeX."""
    latex_df = build_compact_comparison_table(df)
    latex_df["MAE"] = latex_df["MAE"].round(2)
    latex_df["RMSE"] = latex_df["RMSE"].round(2)
    latex_df["R2"] = latex_df["R2"].round(3)
    latex_df["train_time_s"] = latex_df["train_time_s"].round(3)
    latex_df["search_time_s"] = latex_df["search_time_s"].round(3)
    latex_df["inference_time_s"] = latex_df["inference_time_s"].round(6)
    return latex_df


def _maybe_set_log_x(ax, x_values: pd.Series) -> None:
    """Configura escala logarítmica en X cuando los tiempos varían mucho."""
    # Se usa escala log cuando los tiempos difieren en órdenes de magnitud.
    positive_values = x_values[x_values > 0]
    if not positive_values.empty and (positive_values.max() / positive_values.min()) >= 50:
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
        offsets.append(base_offsets[close_count % len(base_offsets)])
    return offsets


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


def _load_with_source(path: Path, approach: str, source_run_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "model" not in df.columns and "model_short" in df.columns:
        df["model"] = df["model_short"]
    df["approach"] = approach
    df["source_run_id"] = source_run_id
    return df


def _combine_and_rank(frames: list[pd.DataFrame], subset: str) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["subset"] = subset.upper()
    if "search_time_s" not in combined.columns:
        combined["search_time_s"] = combined["train_time_s"]
    else:
        combined["search_time_s"] = combined["search_time_s"].fillna(combined["train_time_s"])
    combined["model_short"] = combined["model"].astype(str).apply(shorten_model_name)
    combined = combined.sort_values("RMSE", ascending=True).reset_index(drop=True)
    combined["rank_RMSE"] = combined["RMSE"].rank(method="min", ascending=True).astype(int)
    combined["rank_MAE"] = combined["MAE"].rank(method="min", ascending=True).astype(int)
    combined["rank_R2"] = combined["R2"].rank(method="min", ascending=False).astype(int)
    combined["rank_train_time_s"] = combined["train_time_s"].rank(method="min", ascending=True).astype(int)
    combined["rank_inference_time_s"] = combined["inference_time_s"].rank(method="min", ascending=True).astype(int)
    return combined


# ORQUESTACIÓN PASO 03

def run_paso_03_pipeline(subset: str, baseline_run_id: str, h2o_run_id: str, run_id: str | None = None, output_dir: str = "results") -> dict[str, str]:
    """Ejecuta la comparación inicial y guarda tablas, Pareto y figuras."""
    subset = subset.upper()
    project_root = get_project_root()
    results_root = Path(project_root) / output_dir
    comparison_run_id = run_id or generate_run_id(subset=subset, execution_type="comparison")
    comparison_run_dir = create_run_directory(results_root=results_root, run_id=comparison_run_id)
    clean_directory_contents_for_stable_run(comparison_run_dir, stable_run_ids=STABLE_RUN_IDS)
    figures_dir = comparison_run_dir / "figures"

    baseline_path = results_root / "runs" / baseline_run_id / "baseline_test.csv"
    h2o_path = results_root / "runs" / h2o_run_id / "h2o_test_results.csv"
    frames = [
        _load_with_source(baseline_path, approach="Traditional baseline", source_run_id=baseline_run_id),
        _load_with_source(h2o_path, approach="H2O AutoML", source_run_id=h2o_run_id),
    ]
    combined_df = _combine_and_rank(frames=frames, subset=subset)
    pareto_df = detect_pareto_solutions(combined_df, objectives=["RMSE", "train_time_s", "inference_time_s"])

    comparison_path = comparison_run_dir / "comparison.csv"
    compact_path = comparison_run_dir / "comparison_compact.csv"
    latex_path = comparison_run_dir / "comparison_latex.csv"
    pareto_path = comparison_run_dir / "pareto.csv"

    pareto_df.to_csv(comparison_path, index=False)
    pareto_df[pareto_df["is_pareto_optimal"]].to_csv(pareto_path, index=False)
    build_compact_comparison_table(pareto_df).to_csv(compact_path, index=False)
    build_latex_comparison_table(pareto_df).to_csv(latex_path, index=False)

    create_scatter_figure(
        pareto_df,
        x_col="train_time_s",
        y_col="RMSE",
        title=f"RMSE vs Train Time ({subset})",
        output_path=figures_dir / f"rmse_vs_train_time_{subset}.png",
        label_only_pareto=True,
    )
    create_scatter_figure(
        pareto_df,
        x_col="inference_time_s",
        y_col="RMSE",
        title=f"RMSE vs Inference Time ({subset})",
        output_path=figures_dir / f"rmse_vs_inference_time_{subset}.png",
        label_only_pareto=True,
    )
    create_scatter_figure(
        pareto_df,
        x_col="train_time_s",
        y_col="MAE",
        title=f"MAE vs Train Time ({subset})",
        output_path=figures_dir / f"mae_vs_train_time_{subset}.png",
        label_only_pareto=True,
    )

    config_path = save_run_config(
        comparison_run_dir,
        {
            "run_id": comparison_run_id,
            "subset": subset,
            "execution_type": "comparison",
            "baseline_run_id": baseline_run_id,
            "h2o_run_id": h2o_run_id,
            "nsga2_run_id": None,
            "generated_files": [
                comparison_path.name,
                compact_path.name,
                latex_path.name,
                pareto_path.name,
                "figures",
                "run_config.json",
            ],
        },
    )
    latest_dir = update_latest_files(results_root=results_root, run_type="comparison", source_paths=[comparison_path, compact_path, latex_path, pareto_path, config_path])
    latest_figures_dir = update_latest_comparison_figures(results_root=results_root, figures_dir=figures_dir)
    print(f"Latest comparison copy: {latest_dir}")
    print(f"Latest comparison figures: {latest_figures_dir}")

    return {
        "run_id": comparison_run_id,
        "run_dir": str(comparison_run_dir),
        "comparison_path": str(comparison_path),
        "compact_path": str(compact_path),
        "latex_path": str(latex_path),
        "pareto_path": str(pareto_path),
        "figures_dir": str(figures_dir),
    }


def main() -> None:
    """Ejecuta la configuración final del paso 3: comparación inicial entre modelos tradicionales y H2O AutoML."""
    # Configuración usada en este experimento.
    subset = "FD004"
    baseline_run_id = "FD004_baseline"
    h2o_run_id = "FD004_h2o_30models"
    run_id = "FD004_comparison_initial"
    output_dir = "results"
    outputs = run_paso_03_pipeline(
        subset=subset,
        baseline_run_id=baseline_run_id,
        h2o_run_id=h2o_run_id,
        run_id=run_id,
        output_dir=output_dir,
    )
    print("\n[PASO 03] Ejecutada correctamente")
    print(f"[PASO 03] run_id: {outputs['run_id']}")
    print(f"[PASO 03] run_dir: {outputs['run_dir']}")


if __name__ == "__main__":
    main()
