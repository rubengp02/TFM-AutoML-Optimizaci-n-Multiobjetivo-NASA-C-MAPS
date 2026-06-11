"""PASO 06: comparación final (tradicionales + H2O + NSGA-II)."""

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
    model_text = str(model_name)
    # Los modelos StackedEnsemble_AllModels de H2O se renombran para mejorar legibilidad en tablas y figuras.
    if model_text.startswith("StackedEnsemble_AllModels"):
        return "H2O Stacked Ensemble"
    mapping = {
        "LGBMRegressor": "LightGBM",
        "XGBRegressor": "XGBoost",
        "HistGradientBoostingRegressor": "HistGradientBoosting",
        "RandomForestRegressor": "RandomForest",
        "ExtraTreesRegressor": "ExtraTrees",
        "RidgePipeline": "Ridge",
    }
    return mapping.get(model_text, model_text)


def detect_pareto_solutions(results_df: pd.DataFrame, objectives: list[str]) -> pd.DataFrame:
    """Marca filas no dominadas (Pareto-óptimas) para objetivos de minimización."""
    # Frente de Pareto final sobre error y eficiencia temporal.
    work_df = results_df.copy().reset_index(drop=True)
    objective_values = work_df[objectives].to_numpy(dtype=float)
    pareto_mask = [True] * len(work_df)
    for current_index in range(len(work_df)):
        for comparison_index in range(len(work_df)):
            if current_index == comparison_index:
                continue
            no_worse_all = (objective_values[comparison_index] <= objective_values[current_index]).all()
            strictly_better_one = (objective_values[comparison_index] < objective_values[current_index]).any()
            if no_worse_all and strictly_better_one:
                pareto_mask[current_index] = False
                break
    work_df["is_pareto_optimal"] = pareto_mask
    return work_df


def build_compact_comparison_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Crea una vista compacta de comparación para informes."""
    output_columns = ["model_short", "approach", "MAE", "RMSE", "R2", "train_time_s", "search_time_s", "inference_time_s", "inference_time_per_sample_s", "n_features", "is_pareto_optimal"]
    return comparison_df[output_columns].copy().sort_values("RMSE", ascending=True).reset_index(drop=True)


def build_latex_comparison_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabla redondeada para exportación LaTeX."""
    latex_table_df = build_compact_comparison_table(comparison_df)
    latex_table_df["MAE"] = latex_table_df["MAE"].round(2)
    latex_table_df["RMSE"] = latex_table_df["RMSE"].round(2)
    latex_table_df["R2"] = latex_table_df["R2"].round(3)
    latex_table_df["train_time_s"] = latex_table_df["train_time_s"].round(3)
    latex_table_df["search_time_s"] = latex_table_df["search_time_s"].round(3)
    latex_table_df["inference_time_s"] = latex_table_df["inference_time_s"].round(6)
    return latex_table_df


def _maybe_set_log_x(ax, x_values: pd.Series) -> None:
    """Configura escala logarítmica en X cuando los tiempos varían mucho."""
    # Facilita lectura cuando hay grandes diferencias de tiempo entre modelos.
    positive_values = x_values[x_values > 0]
    if not positive_values.empty and (positive_values.max() / positive_values.min()) >= 50:
        ax.set_xscale("log")


def _compute_label_offsets(dataframe: pd.DataFrame, x_col: str, y_col: str) -> list[tuple[int, int]]:
    """Calcula desplazamientos por punto para reducir solapamiento de etiquetas en puntos cercanos."""
    x_range = max(float(dataframe[x_col].max() - dataframe[x_col].min()), 1e-12)
    y_range = max(float(dataframe[y_col].max() - dataframe[y_col].min()), 1e-12)
    x_tol = 0.08 * x_range
    y_tol = 0.08 * y_range
    base_offsets = [(6, 6), (8, -8), (-10, 7), (7, 11), (-9, -10), (11, 0), (0, 11), (-11, 0)]
    offsets: list[tuple[int, int]] = []
    for row_index, row_i in dataframe.reset_index(drop=True).iterrows():
        close_count = 0
        for previous_index in range(row_index):
            row_j = dataframe.iloc[previous_index]
            if abs(float(row_i[x_col]) - float(row_j[x_col])) <= x_tol and abs(float(row_i[y_col]) - float(row_j[y_col])) <= y_tol:
                close_count += 1
        offsets.append(base_offsets[close_count % len(base_offsets)])
    return offsets


def create_scatter_figure(dataframe: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: Path, label_only_pareto: bool = False) -> None:
    """Crea y guarda una figura de dispersión con mejor legibilidad de etiquetas."""
    figure, axis = plt.subplots(figsize=(10, 6.4))
    axis.scatter(dataframe[x_col], dataframe[y_col])
    _maybe_set_log_x(axis, dataframe[x_col])
    if label_only_pareto and "is_pareto_optimal" in dataframe.columns:
        label_df = dataframe[dataframe["is_pareto_optimal"]].copy().reset_index(drop=True)
    else:
        label_df = dataframe.reset_index(drop=True)
    offsets = _compute_label_offsets(label_df, x_col=x_col, y_col=y_col) if not label_df.empty else []
    for label_index, (_, row) in enumerate(label_df.iterrows()):
        x_offset, y_offset = offsets[label_index]
        axis.annotate(str(row["model_short"]), (row[x_col], row[y_col]), textcoords="offset points", xytext=(x_offset, y_offset), fontsize=8)
    axis.set_xlabel(x_col)
    axis.set_ylabel(y_col)
    axis.set_title(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _load_with_source(file_path: Path, approach: str, source_run_id: str) -> pd.DataFrame:
    loaded_df = pd.read_csv(file_path)
    if "model" not in loaded_df.columns and "model_short" in loaded_df.columns:
        loaded_df["model"] = loaded_df["model_short"]
    if approach == "NSGA-II" and "trial_number" in loaded_df.columns:
        trial_text = loaded_df["trial_number"].astype(float).astype(int).astype(str)
        base_model_text = loaded_df["model_short"] if "model_short" in loaded_df.columns else loaded_df["model"]
        base_model_text = base_model_text.astype(str).str.replace(r"\s+NSGA-II\s+T\d+$", "", regex=True)
        loaded_df["model"] = base_model_text + "_NSGA2_trial_" + trial_text
        loaded_df["model_short"] = base_model_text + " NSGA-II T" + trial_text
    loaded_df["approach"] = approach
    loaded_df["source_run_id"] = source_run_id
    return loaded_df


def _combine_and_rank(dataframes: list[pd.DataFrame], subset: str) -> pd.DataFrame:
    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    combined_df["subset"] = subset.upper()
    if "search_time_s" not in combined_df.columns:
        combined_df["search_time_s"] = combined_df["train_time_s"]
    else:
        combined_df["search_time_s"] = combined_df["search_time_s"].fillna(combined_df["train_time_s"])
    if "model_short" in combined_df.columns:
        combined_df["model_short"] = combined_df["model_short"].fillna("").astype(str)
        blank_mask = combined_df["model_short"].str.strip().eq("")
        combined_df.loc[blank_mask, "model_short"] = combined_df.loc[blank_mask, "model"].astype(str).apply(shorten_model_name)
    else:
        combined_df["model_short"] = combined_df["model"].astype(str).apply(shorten_model_name)
    combined_df = combined_df.sort_values("RMSE", ascending=True).reset_index(drop=True)
    combined_df["rank_RMSE"] = combined_df["RMSE"].rank(method="min", ascending=True).astype(int)
    combined_df["rank_MAE"] = combined_df["MAE"].rank(method="min", ascending=True).astype(int)
    combined_df["rank_R2"] = combined_df["R2"].rank(method="min", ascending=False).astype(int)
    combined_df["rank_train_time_s"] = combined_df["train_time_s"].rank(method="min", ascending=True).astype(int)
    combined_df["rank_inference_time_s"] = combined_df["inference_time_s"].rank(method="min", ascending=True).astype(int)
    return combined_df


# ORQUESTACIÓN PASO 06

def run_paso_06_pipeline(subset: str, baseline_run_id: str, h2o_run_id: str, nsga2_run_id: str, run_id: str | None = None, output_dir: str = "results") -> dict[str, str]:
    """Ejecuta la comparación final integrando baseline, H2O y NSGA-II."""
    subset = subset.upper()
    project_root = get_project_root()
    results_root = Path(project_root) / output_dir
    comparison_run_id = run_id or generate_run_id(subset=subset, execution_type="comparison")
    comparison_run_dir = create_run_directory(results_root=results_root, run_id=comparison_run_id)
    clean_directory_contents_for_stable_run(comparison_run_dir, stable_run_ids=STABLE_RUN_IDS)
    figures_dir = comparison_run_dir / "figures"

    baseline_path = results_root / "runs" / baseline_run_id / "baseline_test.csv"
    h2o_path = results_root / "runs" / h2o_run_id / "h2o_test_results.csv"
    nsga2_path = results_root / "runs" / nsga2_run_id / "nsga2_test_results.csv"
    frames = [
        _load_with_source(baseline_path, approach="Traditional baseline", source_run_id=baseline_run_id),
        _load_with_source(h2o_path, approach="H2O AutoML", source_run_id=h2o_run_id),
        _load_with_source(nsga2_path, approach="NSGA-II", source_run_id=nsga2_run_id),
    ]

    combined_df = _combine_and_rank(dataframes=frames, subset=subset)
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
    pareto_only_df = pareto_df[pareto_df["is_pareto_optimal"]].copy().reset_index(drop=True)
    create_scatter_figure(
        pareto_only_df,
        x_col="train_time_s",
        y_col="RMSE",
        title=f"RMSE vs Train Time Pareto ({subset})",
        output_path=figures_dir / f"rmse_vs_train_time_{subset}_pareto.png",
        label_only_pareto=False,
    )
    create_scatter_figure(
        pareto_only_df,
        x_col="inference_time_s",
        y_col="RMSE",
        title=f"RMSE vs Inference Time Pareto ({subset})",
        output_path=figures_dir / f"rmse_vs_inference_time_{subset}_pareto.png",
        label_only_pareto=False,
    )
    create_scatter_figure(
        pareto_only_df,
        x_col="train_time_s",
        y_col="MAE",
        title=f"MAE vs Train Time Pareto ({subset})",
        output_path=figures_dir / f"mae_vs_train_time_{subset}_pareto.png",
        label_only_pareto=False,
    )

    config_path = save_run_config(
        comparison_run_dir,
        {
            "run_id": comparison_run_id,
            "subset": subset,
            "execution_type": "comparison",
            "baseline_run_id": baseline_run_id,
            "h2o_run_id": h2o_run_id,
            "nsga2_run_id": nsga2_run_id,
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
    """Parsea argumentos CLI y genera la comparación final integrando resultados de NSGA-II."""
    # Configuración usada en este experimento.
    subset = "FD004"
    baseline_run_id = "FD004_baseline"
    h2o_run_id = "FD004_h2o_30models"
    nsga2_run_id = "FD004_nsga2_60trials"
    run_id = "FD004_comparison_final"
    output_dir = "results"
    outputs = run_paso_06_pipeline(
        subset=subset,
        baseline_run_id=baseline_run_id,
        h2o_run_id=h2o_run_id,
        nsga2_run_id=nsga2_run_id,
        run_id=run_id,
        output_dir=output_dir,
    )
    print("\n[PASO 06] Ejecutada correctamente")
    print(f"[PASO 06] run_id: {outputs['run_id']}")
    print(f"[PASO 06] run_dir: {outputs['run_dir']}")


if __name__ == "__main__":
    main()
