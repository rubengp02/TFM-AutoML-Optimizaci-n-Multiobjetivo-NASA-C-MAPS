"""PASO 02: entrenamiento H2O AutoML para C-MAPSS."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import DEFAULT_H2O_MAX_MODELS, DEFAULT_H2O_NFOLDS, RANDOM_STATE
from src.cmapss_data_preparation import get_project_root, prepare_cmapss_data
from src.experiment_tracking import clean_directory_contents_for_stable_run, create_run_directory, generate_run_id, save_run_config, update_latest_files

STABLE_RUN_IDS = {"FD004_baseline", "FD004_h2o_30models", "FD004_comparison_initial", "FD004_compromise_selection", "FD004_nsga2_60trials", "FD004_comparison_final"}


# COMPROBACIÓN E INICIALIZACIÓN H2O

try:
    import h2o
    from h2o.automl import H2OAutoML
except ImportError as exc:  # pragma: no cover
    h2o = None
    H2OAutoML = None
    _H2O_IMPORT_ERROR = exc
else:
    _H2O_IMPORT_ERROR = None


def ensure_h2o_installed() -> None:
    """Lanza un error claro si H2O no está instalado."""
    if _H2O_IMPORT_ERROR is not None:
        raise ImportError("H2O is not installed. Install it with: pip install h2o") from _H2O_IMPORT_ERROR


def initialize_h2o(max_mem_size: str | None = "3G", nthreads: int = 4) -> None:
    """Inicializa H2O de forma segura y evita problemas de arranque repetidos."""
    ensure_h2o_installed()
    if h2o.cluster() is None:
        h2o.init(max_mem_size=max_mem_size, nthreads=nthreads)
    else:
        h2o.no_progress()


# PREPARACIÓN Y ENTRENAMIENTO H2O

def pandas_to_h2o_frame(dataframe: pd.DataFrame, frame_name: str | None = None):
    """Convierte un DataFrame de pandas en un H2OFrame."""
    # H2O utiliza su propia clase H2OFrame y no entrena directamente sobre DataFrames de pandas.
    return h2o.H2OFrame(dataframe, destination_frame=frame_name)


def get_h2o_feature_columns(train_df: pd.DataFrame) -> list[str]:
    """Devuelve las columnas predictoras usadas por H2O AutoML.
    
    Se excluyen:
    - unit: identificador del motor, no debe usarse como variable predictora.
    - RUL: variable objetivo que el modelo debe predecir.
    """
    # Se mantiene `cycle` y se excluye `unit` por ser identificador de motor.
    return [column_name for column_name in train_df.columns if column_name not in {"unit", "RUL"}]


def build_default_project_name(subset: str, max_models: int, seed: int, suffix: str = "final") -> str:
    """Construye un nombre descriptivo para una ejecución de H2O AutoML."""
    # El nombre se construye por partes para que sea trazable: subset, h2o, max_models, seed y sufijo.
    subset_norm = subset.upper()
    return "_".join([f"cmapss_{subset_norm}", "h2o", f"maxmodels_{max_models}", f"seed_{seed}", suffix])


def get_h2o_leader_train_time_seconds(leaderboard_df: pd.DataFrame, leader_model_id: str, search_time_s: float) -> tuple[float, str]:
    """Devuelve el train_time_s del líder desde training_time_ms del leaderboard con fallback a search_time_s."""
    if "training_time_ms" not in leaderboard_df.columns:
        return float(search_time_s), "fallback_search_time_s"

    leader_rows = leaderboard_df[leaderboard_df["model_id"].astype(str) == str(leader_model_id)]
    if leader_rows.empty:
        return float(search_time_s), "fallback_search_time_s"

    training_time_ms = pd.to_numeric(leader_rows.iloc[0]["training_time_ms"], errors="coerce")
    if pd.isna(training_time_ms):
        return float(search_time_s), "fallback_search_time_s"

    return float(training_time_ms) / 1000.0, "leaderboard_training_time_ms"


def train_h2o_automl_regressor(train_df: pd.DataFrame, subset: str, max_models: int = DEFAULT_H2O_MAX_MODELS, max_runtime_secs: int | None = None, seed: int = RANDOM_STATE) -> dict[str, Any]:
    """Entrena H2O AutoML para regresión de RUL y devuelve artefactos de entrenamiento."""
    ensure_h2o_installed()
    subset_norm = subset.upper()
    features = get_h2o_feature_columns(train_df)
    # Creación del nombre del proyecto para esta ejecución.
    run_project_name = build_default_project_name(
        subset=subset_norm,
        max_models=max_models,
        seed=seed,
        suffix="final",
    )
    # Se convierte el DataFrame a H2OFrame.
    train_h2o = pandas_to_h2o_frame(
        train_df,
        frame_name=f"train_{subset_norm}",
    )

    # Se configura y entrena el modelo H2O AutoML.
    automl_model = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        nfolds=DEFAULT_H2O_NFOLDS,
        seed=seed,
        # Necesario para que H2O construya Stacked Ensembles de forma estable.
        keep_cross_validation_predictions=True,
        sort_metric="RMSE",
        project_name=run_project_name,
    )

    # Se mide el tiempo de entrenamiento.
    training_start_time = perf_counter()
    automl_model.train(x=features, y="RUL", training_frame=train_h2o)
    search_time_s = perf_counter() - training_start_time
    train_time_s = float(search_time_s)
    # Se obtiene el leaderboard como DataFrame de pandas para su posterior análisis.
    leaderboard_df = automl_model.leaderboard.as_data_frame()
    try:
        expanded_leaderboard = h2o.automl.get_leaderboard(automl_model, extra_columns="ALL")
        leaderboard_df = expanded_leaderboard.as_data_frame()
    except Exception:
        pass

    return {
        "subset": subset_norm,
        "run_project_name": run_project_name,
        "features": features,
        "leader": automl_model.leader,
        "leaderboard": leaderboard_df,
        "train_time_s": train_time_s,
        "search_time_s": float(search_time_s),
        "automl": automl_model,
    }


def evaluate_h2o_leader_on_test(leader, test_df: pd.DataFrame, y_test: pd.Series, feature_cols: list[str]) -> dict[str, Any]:
    """Ejecuta inferencia del líder y calcula métricas de sklearn en el test oficial."""
    ensure_h2o_installed()
    # El test oficial solo se utiliza aquí para evaluación final del líder.
    evaluation_df = test_df.copy()
    evaluation_df["RUL"] = y_test.reset_index(drop=True)
    test_h2o = pandas_to_h2o_frame(evaluation_df, frame_name="test_official_eval")

    inference_start_time = perf_counter()
    h2o_predictions = leader.predict(test_h2o)
    inference_time_s = perf_counter() - inference_start_time

    y_pred = h2o_predictions.as_data_frame().iloc[:, 0].to_numpy(dtype=float)
    y_true = y_test.to_numpy(dtype=float)

    # Cálculo de métricas homogéneo con los modelos tradicionales.
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
        "inference_time_s": float(inference_time_s),
        "inference_time_per_sample_s": float(inference_time_s / len(evaluation_df)),
        "n_features": int(len(feature_cols)),
        "predictions": y_pred,
    }


# EXPORTACIÓN DE HIPERPARÁMETROS H2O

def export_h2o_leader_hyperparameters(leader, subset: str) -> pd.DataFrame:
    """Extrae los hiperparámetros del mejor modelo de H2O AutoML."""
    subset_norm = subset.upper()
    model_id = str(leader.model_id)
    # Lista donde se guardará una fila por cada hiperparámetro.
    hyperparameter_rows = []
    # Se recorre el diccionario de hiperparámetros del modelo líder.
    for parameter_name, parameter_info in leader.params.items():
        hyperparameter_rows.append(
            {
                "subset": subset_norm,
                "model_id": model_id,
                "parameter": str(parameter_name),
                "value": str(parameter_info.get("actual")),
            }
        )
    return pd.DataFrame(hyperparameter_rows, columns=["subset", "model_id", "parameter", "value"])


def export_h2o_all_models_hyperparameters(leaderboard_df: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Extrae los hiperparámetros usados por todos los modelos del leaderboard de H2O."""
    ensure_h2o_installed()
    subset_norm = subset.upper()
    # Cada fila almacenará un hiperparámetro concreto de un modelo concreto.
    hyperparameter_rows = []
    # El leaderboard contiene los identificadores de todos los modelos entrenados por H2O AutoML.
    for model_id in leaderboard_df["model_id"].astype(str).tolist():
        # H2O guarda los parámetros de cada modelo en model.params.
        model = h2o.get_model(model_id)
        for parameter_name, parameter_info in model.params.items():
            hyperparameter_rows.append(
                {
                    "subset": subset_norm,
                    "model_id": model_id,
                    "parameter": str(parameter_name),
                    "value": str(parameter_info.get("actual")),
                }
            )
    return pd.DataFrame(hyperparameter_rows, columns=["subset", "model_id", "parameter", "value"])


# ORQUESTACIÓN DE PASO 02

def run_paso_02_pipeline(subset: str, max_models: int = DEFAULT_H2O_MAX_MODELS, max_runtime_secs: int | None = None, run_id: str | None = None, output_dir: str = "results") -> dict[str, str]:
    """Ejecuta el pipeline completo del PASO 02 y guarda sus artefactos."""
    subset = subset.upper()
    project_root = get_project_root()
    results_root = Path(project_root) / output_dir
    run_id = run_id or generate_run_id(subset=subset, execution_type="h2o", main_descriptor=f"{max_models}models")
    run_dir = create_run_directory(results_root=results_root, run_id=run_id)
    clean_directory_contents_for_stable_run(run_dir, stable_run_ids=STABLE_RUN_IDS)

    print(f"[INFO] Loading C-MAPSS subset: {subset}")
    data_bundle = prepare_cmapss_data(subset=subset, raw_dir=project_root / "data" / "raw")
    train_df = data_bundle["train_df"]
    test_last_df = data_bundle["test_last_df"]
    y_test = data_bundle["y_test"]

    initialize_h2o()
    training = train_h2o_automl_regressor(
        train_df=train_df,
        subset=subset,
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        seed=RANDOM_STATE,
    )

    leaderboard_df = training["leaderboard"]
    leader = training["leader"]
    feature_cols = training["features"]
    run_project_name = str(training["run_project_name"])
    leaderboard_df["run_project_name"] = run_project_name
    h2o_leader_model_id = str(leader.model_id)
    leader_train_time_s, train_time_source = get_h2o_leader_train_time_seconds(
        leaderboard_df=leaderboard_df,
        leader_model_id=h2o_leader_model_id,
        search_time_s=float(training["search_time_s"]),
    )

    test_metrics = evaluate_h2o_leader_on_test(
        leader=leader,
        test_df=test_last_df,
        y_test=y_test,
        feature_cols=feature_cols,
    )
    summary_df = pd.DataFrame(
        [
            {
                "model": leader.model_id,
                "MAE": test_metrics["MAE"],
                "RMSE": test_metrics["RMSE"],
                "R2": test_metrics["R2"],
                "train_time_s": leader_train_time_s,
                "search_time_s": training["search_time_s"],
                "train_time_source": train_time_source,
                "h2o_leader_model_id": h2o_leader_model_id,
                "inference_time_s": test_metrics["inference_time_s"],
                "inference_time_per_sample_s": test_metrics["inference_time_per_sample_s"],
                "n_features": test_metrics["n_features"],
                "automl_tool": "H2OAutoML",
                "run_project_name": run_project_name,
            }
        ]
    )

    leaderboard_path = run_dir / "h2o_leaderboard.csv"
    results_path = run_dir / "h2o_test_results.csv"
    hparams_leader_path = run_dir / "hyperparameters_h2o_leader.csv"
    hparams_all_path = run_dir / "hyperparameters_h2o_all_models.csv"

    leaderboard_df.to_csv(leaderboard_path, index=False)
    summary_df.to_csv(results_path, index=False)
    export_h2o_leader_hyperparameters(
        leader=leader,
        subset=subset,
    ).to_csv(hparams_leader_path, index=False)
    export_h2o_all_models_hyperparameters(
        leaderboard_df=leaderboard_df,
        subset=subset,
    ).to_csv(hparams_all_path, index=False)

    config_path = save_run_config(
        run_dir,
        {
            "run_id": run_id,
            "subset": subset,
            "execution_type": "h2o",
            "max_models": max_models,
            "max_runtime_secs": max_runtime_secs,
            "seed": RANDOM_STATE,
            "project_name": run_project_name,
            "keep_cross_validation_predictions": True,
            "nfolds": DEFAULT_H2O_NFOLDS,
            "run_project_name": run_project_name,
            "train_time_equals_search_time": bool(train_time_source == "fallback_search_time_s"),
            "automl_search_time_s": float(training["search_time_s"]),
            "leader_train_time_s": float(leader_train_time_s),
            "leader_train_time_source": train_time_source,
            "h2o_leader_model_id": h2o_leader_model_id,
            "generated_files": [
                leaderboard_path.name,
                results_path.name,
                hparams_leader_path.name,
                hparams_all_path.name,
                "run_config.json",
            ],
        },
    )
    latest_dir = update_latest_files(
        results_root=results_root,
        run_type="h2o",
        source_paths=[
            leaderboard_path,
            results_path,
            hparams_leader_path,
            hparams_all_path,
            config_path,
        ],
    )

    print(f"\n[INFO] Project name used: {run_project_name}")
    print(f"[INFO] Number of models in leaderboard: {len(leaderboard_df)}")
    print(f"[INFO] Leader model_id: {leader.model_id}")
    print(f"[INFO] Latest H2O copy: {latest_dir}")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "leaderboard_path": str(leaderboard_path),
        "results_path": str(results_path),
        "hparams_leader_path": str(hparams_leader_path),
        "hparams_all_path": str(hparams_all_path),
        "config_path": str(config_path),
    }


def main() -> None:
    """Parsea argumentos CLI y ejecuta H2O AutoML, guardando leaderboard y evaluación en test oficial."""
    # Configuración usada en este experimento.
    subset = "FD004"
    max_models = 30
    max_runtime_secs = None
    run_id = "FD004_h2o_30models"
    output_dir = "results"
    outputs = run_paso_02_pipeline(
        subset=subset,
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        run_id=run_id,
        output_dir=output_dir,
    )
    print("\n[PASO 02] Ejecutada correctamente")
    print(f"[PASO 02] run_id: {outputs['run_id']}")
    print(f"[PASO 02] run_dir: {outputs['run_dir']}")


if __name__ == "__main__":
    main()

