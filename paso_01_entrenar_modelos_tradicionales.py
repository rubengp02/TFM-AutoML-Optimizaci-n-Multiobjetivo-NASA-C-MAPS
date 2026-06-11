"""PASO 01: entrenamiento de modelos tradicionales para C-MAPSS."""

# IMPORTS

from __future__ import annotations

import warnings
import itertools
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_VALIDATION_SIZE, LOW_VARIANCE_THRESHOLD, RANDOM_STATE
from src.cmapss_data_preparation import get_project_root, prepare_cmapss_data
from src.experiment_tracking import clean_directory_contents_for_stable_run, create_run_directory, generate_run_id, save_run_config, update_latest_files

MAX_RANDOM_SEARCH_CONFIGS = 20
STABLE_RUN_IDS = {"FD004_baseline", "FD004_h2o_30models", "FD004_comparison_initial", "FD004_compromise_selection", "FD004_nsga2_60trials", "FD004_comparison_final"}


# EXPLORACIÓN BÁSICA

def get_dimensions(train_df: pd.DataFrame, test_df: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Devuelve dimensiones de train, test e y_test."""
    return pd.DataFrame(
        [
            {"dataset": "train", "rows": train_df.shape[0], "cols": train_df.shape[1]},
            {"dataset": "test", "rows": test_df.shape[0], "cols": test_df.shape[1]},
            {"dataset": "y_test", "rows": y_test.shape[0], "cols": 1},
        ]
    )


def count_engines(df: pd.DataFrame) -> int:
    """Cuenta motores únicos (units) en un conjunto de datos."""
    return int(df["unit"].nunique())


def cycles_per_engine_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resume la distribución del número de ciclos por motor."""

    # Agrupa por motor y cuenta cuántos ciclos tiene cada motor.
    cycles_per_engine = df.groupby("unit")["cycle"].count().rename("cycle_count")
    return cycles_per_engine.describe().to_frame().T


def null_values_summary(df: pd.DataFrame) -> pd.Series:
    """Devuelve el número de nulos por columna."""
    return df.isnull().sum()


def low_variance_columns(df: pd.DataFrame, threshold: float = LOW_VARIANCE_THRESHOLD) -> pd.Series:
    """Detecta columnas numéricas con varianza menor o igual al umbral."""
    numeric_df = df.select_dtypes(include=[np.number])
    variances = numeric_df.var(numeric_only=True)
    return variances[variances <= threshold].sort_values()


def operational_sensor_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve estadísticas descriptivas de ajustes operativos y sensores."""
    operational_columns = []
    for column_name in df.columns:
        if column_name.startswith("op_setting_"):
            operational_columns.append(column_name)

    sensor_columns = []
    for column_name in df.columns:
        if column_name.startswith("sensor_"):
            sensor_columns.append(column_name)

    selected_columns = operational_columns + sensor_columns
    return df[selected_columns].describe().T


def run_basic_exploration(train_df: pd.DataFrame, test_df: pd.DataFrame, y_test: pd.Series, low_var_threshold: float = LOW_VARIANCE_THRESHOLD) -> dict[str, pd.DataFrame | pd.Series | int]:
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


# MODELOS TRADICIONALES

def build_ridge_pipeline() -> Pipeline:
    """Construye Ridge con escalado como modelo lineal de referencia."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def build_random_forest(random_state: int) -> RandomForestRegressor:
    """Construye RandomForest como baseline de árboles (bagging)."""
    return RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=4,
    )


def build_extra_trees(random_state: int) -> ExtraTreesRegressor:
    """Construye ExtraTrees como baseline de árboles altamente aleatorizados."""
    return ExtraTreesRegressor(
        n_estimators=400,
        random_state=random_state,
        n_jobs=4,
    )


def build_hist_gradient_boosting(random_state: int) -> HistGradientBoostingRegressor:
    """Construye HistGradientBoosting como baseline boosting para tabular."""
    return HistGradientBoostingRegressor(
        random_state=random_state,
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
    )


def build_xgboost(random_state: int):
    """Construye XGBoost si la librería está disponible."""
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=4,
    )


def build_lightgbm(random_state: int):
    """Construye LightGBM si la librería está disponible."""
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=4,
    )


def build_baseline_models(random_state: int = RANDOM_STATE) -> dict[str, object]:
    """Construye el conjunto explícito de modelos tradicionales a usar.
    
    Grupos metodológicos:
    - Ridge: referencia lineal.
    - RandomForest / ExtraTrees: métodos basados en árboles.
    - HistGradientBoosting / XGBoost / LightGBM: métodos boosting tabulares.
    """
    
    # Creamos un diccionario donde las claves son los nombres de los modelos y los valores son los objetos construidos.
    models = {
        "Ridge": build_ridge_pipeline(),
        "RandomForest": build_random_forest(random_state=random_state),
        "ExtraTrees": build_extra_trees(random_state=random_state),
        "HistGradientBoosting": build_hist_gradient_boosting(random_state=random_state),
    }
    # XGBoost y LightGBM se incluyen si están instalados, pero no son obligatorios.
    try:
        models["XGBoost"] = build_xgboost(random_state=random_state)
    except ImportError:
        warnings.warn("xgboost no está instalado. Se omite XGBoost.", stacklevel=2)
    try:
        models["LightGBM"] = build_lightgbm(random_state=random_state)
    except ImportError:
        warnings.warn("lightgbm no está instalado. Se omite LightGBM.", stacklevel=2)
    return models


def print_baseline_model_summary(models: dict[str, object]) -> None:
    """Imprime un resumen visual de modelos tradicionales a usar."""
    print("\nModelos tradicionales utilizados:")
    for model_name in models.keys():
        print(f"- {model_name}")


def get_feature_columns(train_df: pd.DataFrame, drop_low_variance: bool = True, low_var_threshold: float = LOW_VARIANCE_THRESHOLD) -> tuple[list[str], list[str]]:
    """Devuelve las columnas de entrada y las columnas eliminadas por baja varianza.
    
    La varianza baja se calcula solo en entrenamiento para evitar fuga de información.
    """
    # Se excluyen el identificador del motor y la variable objetivo.
    columns_to_exclude = {"RUL", "unit"}
    base_features = []
    for column_name in train_df.columns:
        if column_name not in columns_to_exclude:
            base_features.append(column_name)
    # Si no se aplica el filtro de baja varianza, se usan todas las variables base.
    if not drop_low_variance:
        return base_features, []
    # Se calcula la varianza solo en las variables numéricas del entrenamiento.
    numeric_variances = train_df[base_features].select_dtypes(
        include=[np.number]
    ).var(numeric_only=True)
    low_variance_feature_columns = numeric_variances[numeric_variances <= low_var_threshold].index.tolist()
    selected_feature_columns = []
    for column_name in base_features:
        if column_name not in low_variance_feature_columns:
            selected_feature_columns.append(column_name)

    return selected_feature_columns, sorted(low_variance_feature_columns)


def split_train_validation_by_unit(train_df: pd.DataFrame, validation_size: float = DEFAULT_VALIDATION_SIZE, random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide train según motores completos (unit), no por filas, ya que podríamos acabar con ciclos del mismo motor en entrenamiento 
    y en validación si no se hiciera.
    
    Esto evita mezclar ciclos del mismo motor entre train y validación.
    """
    unit_ids = train_df["unit"].drop_duplicates().values
    train_units, validation_units = train_test_split(
        unit_ids,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )
    # Se seleccionan las filas correspondientes a los IDs de entrenamiento y validación respectivamente, asegurando que cada motor completo ("unit") esté solo en uno de los conjuntos.
    train_split = train_df[train_df["unit"].isin(train_units)].copy()
    val_split = train_df[train_df["unit"].isin(validation_units)].copy()
    return train_split, val_split


def _evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas de regresión: MAE, RMSE y R2."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _fit_predict_timed(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """Ajusta modelo y devuelve predicciones junto con tiempos de entrenamiento e inferencia."""
    # Se mide el tiempo de entrenamiento.
    train_start = perf_counter()
    model.fit(X_train, y_train)
    train_time = perf_counter() - train_start
    # Se mide el tiempo de inferencia.
    inference_start = perf_counter()
    y_pred = model.predict(X_eval)
    inference_time = perf_counter() - inference_start
    return y_pred, train_time, inference_time


def evaluate_single_model(model_name: str, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series, feature_columns: list[str], removed_low_variance_columns: list[str]) -> dict[str, float | str | int]:
    """Entrena un modelo, mide tiempos y calcula métricas en el conjunto de evaluación."""
    # Obtenemos predicciones y tiempos de entrenamiento e inferencia.
    y_pred, train_time, inference_time = _fit_predict_timed(model, X_train, y_train, X_eval)
    # Se calculan las métricas de evaluación.
    metrics = _evaluate_predictions(y_eval, y_pred)
    # Se crea un string con las columnas eliminadas, separadas por ";", para que sea fácil almacenar en el CSV de resultados.
    removed_columns_text = ";".join(removed_low_variance_columns)

    # Se devuelve un diccionario con el nombre del modelo, métricas, tiempos, número de características y columnas eliminadas.
    return {
        "model": model_name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "train_time_s": float(train_time),
        "inference_time_s": float(inference_time),
        "inference_time_per_sample_s": float(inference_time / len(X_eval)),
        "n_features": int(len(feature_columns)),
        "removed_low_variance_columns": removed_columns_text,
    }


def evaluate_models(models: dict[str, object], train_df: pd.DataFrame, eval_df: pd.DataFrame, eval_target: pd.Series, feature_columns: list[str], removed_low_variance_columns: list[str]) -> pd.DataFrame:
    """Entrena todos los modelos con train_df y los evalúa sobre eval_df.
    
    El conjunto de evaluación puede ser validación interna o test oficial.
    No se utiliza para ajustar los modelos.
    """
    # Se separan las características y target en train y validation.
    X_train = train_df[feature_columns]
    y_train = train_df["RUL"]
    X_eval = eval_df[feature_columns]
    # Se crea una lista de diccionarios, donde se guarda una fila de resultados por cada modelo evaluado.
    result_rows = []
    for model_name, model in models.items():
        result_rows.append(
            evaluate_single_model(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_eval=X_eval,
                y_eval=eval_target,
                feature_columns=feature_columns,
                removed_low_variance_columns=removed_low_variance_columns,
            )
        )
    # Se devuelve un DataFrame con los resultados de todos los modelos, ordenado por RMSE de mejor a peor.
    return pd.DataFrame(result_rows).sort_values("RMSE", ascending=True).reset_index(drop=True)


def collect_model_hyperparameters(models: dict[str, object], subset: str) -> pd.DataFrame:
    """Extrae hiperparámetros con get_params() en formato largo y se guarda en un DataFrame para posterior análisis."""
    subset_norm = subset.upper()
    hyperparameter_rows = []
    # Se itera sobre cada modelo del diccionario.
    for model_name, model in models.items():
        model_parameters = model.get_params(deep=True)
        # Se itera sobre los hiperparámetros y sus valores.
        for parameter_name, value in model_parameters.items():
            hyperparameter_rows.append(
                {
                    "subset": str(subset_norm),
                    "model": str(model_name),
                    "parameter": str(parameter_name),
                    "value": str(value),
                }
            )
    # Se devuelve un dataframe con los hiperparámetros de todos los modelos y con las siguientes columnas: subset, model, parameter, value.
    return pd.DataFrame(hyperparameter_rows, columns=["subset", "model", "parameter", "value"])


def save_results(results_df: pd.DataFrame, output_path: Path) -> Path:
    """Guarda resultados en CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return output_path


def save_low_variance_columns(removed_columns: list[str], output_path: Path) -> Path:
    """Guarda columnas eliminadas por baja varianza en un CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"column": removed_columns}).to_csv(output_path, index=False)
    return output_path


def save_baseline_hyperparameters(hyperparameters_df: pd.DataFrame, output_path: Path) -> Path:
    """Guarda hiperparámetros baseline en CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hyperparameters_df.to_csv(output_path, index=False)
    return output_path


def get_random_search_spaces() -> dict[str, dict[str, list[Any]]]:
    return {
        "Ridge": {
            "alpha": [0.1, 1.0, 10.0, 50.0],
        },
        "RandomForest": {
            "n_estimators": [200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 3],
        },
        "ExtraTrees": {
            "n_estimators": [300, 400],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 3],
        },
        "HistGradientBoosting": {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_iter": [200, 300],
            "max_depth": [4, 6, 8],
            "l2_regularization": [0.0, 0.1],
        },
        "XGBoost": {
            "n_estimators": [300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [4, 6],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
        },
        "LightGBM": {
            "n_estimators": [300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "min_child_samples": [10, 20],
        },
    }


def build_model_from_config(model_name: str, config: dict[str, Any], random_state: int) -> Any:
    if model_name == "Ridge":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=float(config["alpha"]))),
            ]
        )
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=int(config["n_estimators"]),
            max_depth=config["max_depth"],
            min_samples_leaf=int(config["min_samples_leaf"]),
            random_state=random_state,
            n_jobs=4,
        )
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=int(config["n_estimators"]),
            max_depth=config["max_depth"],
            min_samples_leaf=int(config["min_samples_leaf"]),
            random_state=random_state,
            n_jobs=4,
        )
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            learning_rate=float(config["learning_rate"]),
            max_iter=int(config["max_iter"]),
            max_depth=int(config["max_depth"]),
            l2_regularization=float(config["l2_regularization"]),
            random_state=random_state,
        )
    if model_name == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=int(config["n_estimators"]),
            learning_rate=float(config["learning_rate"]),
            max_depth=int(config["max_depth"]),
            subsample=float(config["subsample"]),
            colsample_bytree=float(config["colsample_bytree"]),
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=4,
        )
    if model_name == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=int(config["n_estimators"]),
            learning_rate=float(config["learning_rate"]),
            num_leaves=int(config["num_leaves"]),
            min_child_samples=int(config["min_child_samples"]),
            random_state=random_state,
            n_jobs=4,
        )
    raise ValueError(f"Modelo no soportado en búsqueda: {model_name}")


def expand_config_space(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    parameter_names = list(space.keys())
    parameter_values = []
    for parameter_name in parameter_names:
        parameter_values.append(space[parameter_name])
    expanded = []
    for values in itertools.product(*parameter_values):
        config = {}
        for idx, parameter_name in enumerate(parameter_names):
            config[parameter_name] = values[idx]
        expanded.append(config)
    return expanded


def sample_configurations(configurations: list[dict[str, Any]], max_configs: int, random_state: int, model_name: str) -> list[dict[str, Any]]:
    if len(configurations) <= max_configs:
        return configurations
    rng_seed = random_state + sum(ord(char) for char in model_name)
    rng = np.random.RandomState(rng_seed)
    indices = np.arange(len(configurations))
    rng.shuffle(indices)
    selected_indices = indices[:max_configs]
    sampled = []
    for selected_index in selected_indices:
        sampled.append(configurations[int(selected_index)])
    return sampled


def evaluate_search_config(model_name: str, config: dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series, feature_columns: list[str], removed_low_variance_columns: list[str], random_state: int) -> dict[str, Any]:
    model = build_model_from_config(model_name, config, random_state=random_state)
    row = evaluate_single_model(
        model_name=model_name,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        feature_columns=feature_columns,
        removed_low_variance_columns=removed_low_variance_columns,
    )
    row["params_json"] = json.dumps(config, sort_keys=True)
    return row


def run_limited_random_search(models: dict[str, object], train_df: pd.DataFrame, validation_df: pd.DataFrame, feature_columns: list[str], removed_low_variance_columns: list[str], random_state: int, max_configs: int = MAX_RANDOM_SEARCH_CONFIGS) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame, dict[str, float], dict[str, int]]:
    X_train = train_df[feature_columns]
    y_train = train_df["RUL"]
    X_validation = validation_df[feature_columns]
    y_validation = validation_df["RUL"]

    search_spaces = get_random_search_spaces()
    search_rows = []
    best_rows = []
    best_configs: dict[str, dict[str, Any]] = {}
    search_time_by_model: dict[str, float] = {}
    n_configs_by_model: dict[str, int] = {}

    for model_name in models.keys():
        if model_name not in search_spaces:
            continue
        full_configurations = expand_config_space(search_spaces[model_name])
        sampled_configurations = sample_configurations(
            full_configurations,
            max_configs=max_configs,
            random_state=random_state,
            model_name=model_name,
        )
        model_search_start_time = perf_counter()
        model_search_rows_start_index = len(search_rows)
        best_row_for_model: dict[str, Any] | None = None
        best_rmse_for_model = float("inf")
        n_configs_evaluated = 0
        for config in sampled_configurations:
            n_configs_evaluated += 1
            config_row = evaluate_search_config(
                model_name=model_name,
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_eval=X_validation,
                y_eval=y_validation,
                feature_columns=feature_columns,
                removed_low_variance_columns=removed_low_variance_columns,
                random_state=random_state,
            )
            config_row["search_strategy"] = "limited_random_search"
            config_row["selected_by"] = "RMSE_validation"
            config_row["n_configs_evaluated"] = len(sampled_configurations)
            search_rows.append(config_row)
            current_rmse = float(config_row["RMSE"])
            if current_rmse < best_rmse_for_model:
                best_rmse_for_model = current_rmse
                best_row_for_model = dict(config_row)
                best_configs[model_name] = dict(config)
        model_search_time_s = perf_counter() - model_search_start_time
        search_time_by_model[model_name] = float(model_search_time_s)
        n_configs_by_model[model_name] = int(n_configs_evaluated)
        for row_index in range(model_search_rows_start_index, len(search_rows)):
            search_rows[row_index]["search_time_s"] = float(model_search_time_s)
        if best_row_for_model is not None:
            best_row_for_model["n_configs_evaluated"] = n_configs_evaluated
            best_row_for_model["search_time_s"] = float(model_search_time_s)
            best_rows.append(best_row_for_model)

    search_results_df = pd.DataFrame(search_rows).sort_values(["model", "RMSE"], ascending=[True, True]).reset_index(drop=True)
    validation_results_df = pd.DataFrame(best_rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    return validation_results_df, best_configs, search_results_df, search_time_by_model, n_configs_by_model


def evaluate_selected_configs_on_test(best_configs: dict[str, dict[str, Any]], search_time_by_model: dict[str, float], n_configs_by_model: dict[str, int], train_df: pd.DataFrame, test_df: pd.DataFrame, y_test: pd.Series, feature_columns: list[str], removed_low_variance_columns: list[str], random_state: int) -> tuple[pd.DataFrame, dict[str, object]]:
    X_train = train_df[feature_columns]
    y_train = train_df["RUL"]
    X_test = test_df[feature_columns]
    selected_models = {}
    test_rows = []
    for model_name, config in best_configs.items():
        model = build_model_from_config(model_name, config, random_state=random_state)
        selected_models[model_name] = model
        row = evaluate_single_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_test,
            y_eval=y_test,
            feature_columns=feature_columns,
            removed_low_variance_columns=removed_low_variance_columns,
        )
        row["search_strategy"] = "limited_random_search"
        row["selected_by"] = "RMSE_validation"
        row["n_configs_evaluated"] = int(n_configs_by_model.get(model_name, 0))
        row["search_time_s"] = float(search_time_by_model.get(model_name, np.nan))
        test_rows.append(row)
    test_results_df = pd.DataFrame(test_rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    return test_results_df, selected_models


# ORQUESTACIÓN DE PASO 01

def print_exploration_summary(subset: str, exploration: dict) -> None:
    """Imprime un resumen de exploración básica para el subset cargado."""
    print(f"\n=== {subset} EXPLORACIÓN BÁSICA ===")
    print("\nDimensiones:")
    print(exploration["dimensions"].to_string(index=False))
    print(f"\nMotores (train): {exploration['n_engines_train']}")
    print(f"Motores (test): {exploration['n_engines_test']}")
    print("\nResumen del número de ciclos - train:")
    print(exploration["cycles_train"].to_string(index=False))
    print("\nResumen del número de ciclos - test:")
    print(exploration["cycles_test"].to_string(index=False))


def run_paso_01_pipeline(subset: str, validation_size: float = DEFAULT_VALIDATION_SIZE, drop_low_variance: bool = True, run_id: str | None = None) -> dict[str, str]:
    """Ejecuta el pipeline completo del PASO 01 y guarda sus artefactos."""
    subset = subset.upper()
    project_root = get_project_root()
    results_root = Path(project_root) / "results"
    run_id = run_id or generate_run_id(subset=subset, execution_type="baseline")
    run_dir = create_run_directory(results_root=results_root, run_id=run_id)
    clean_directory_contents_for_stable_run(run_dir, stable_run_ids=STABLE_RUN_IDS)

    data_bundle = prepare_cmapss_data(subset=subset, raw_dir=project_root / "data" / "raw")
    train_df = data_bundle["train_df"]
    test_df = data_bundle["test_df"]
    test_last_df = data_bundle["test_last_df"]
    y_test = data_bundle["y_test"]

    exploration = run_basic_exploration(train_df, test_df, y_test)
    print_exploration_summary(subset, exploration)

    train_split_df, validation_split_df = split_train_validation_by_unit(
        train_df,
        validation_size=validation_size,
        random_state=RANDOM_STATE,
    )
    feature_columns, removed_low_variance_columns = get_feature_columns(
        train_split_df,
        drop_low_variance=drop_low_variance,
    )
    available_models = build_baseline_models(random_state=RANDOM_STATE)
    print_baseline_model_summary(available_models)

    validation_results, best_configs, search_results, search_time_by_model, n_configs_by_model = run_limited_random_search(
        models=available_models,
        train_df=train_split_df,
        feature_columns=feature_columns,
        removed_low_variance_columns=removed_low_variance_columns,
        validation_df=validation_split_df,
        random_state=RANDOM_STATE,
        max_configs=MAX_RANDOM_SEARCH_CONFIGS,
    )

    test_results, selected_models = evaluate_selected_configs_on_test(
        best_configs=best_configs,
        search_time_by_model=search_time_by_model,
        n_configs_by_model=n_configs_by_model,
        train_df=train_df,
        test_df=test_last_df,
        y_test=y_test,
        feature_columns=feature_columns,
        removed_low_variance_columns=removed_low_variance_columns,
        random_state=RANDOM_STATE,
    )

    validation_path = run_dir / "baseline_validation.csv"
    test_path = run_dir / "baseline_test.csv"
    low_variance_path = run_dir / "low_variance_columns.csv"
    hyperparameters_path = run_dir / "hyperparameters_baseline.csv"
    search_results_path = run_dir / "baseline_search_results.csv"

    save_results(validation_results, validation_path)
    save_results(test_results, test_path)
    save_results(search_results, search_results_path)
    save_low_variance_columns(removed_low_variance_columns, low_variance_path)
    baseline_hyperparameters_df = collect_model_hyperparameters(
        selected_models,
        subset=subset,
    )
    save_baseline_hyperparameters(
        baseline_hyperparameters_df,
        hyperparameters_path,
    )

    config_path = save_run_config(
        run_dir,
        {
            "run_id": run_id,
            "subset": subset,
            "execution_type": "baseline",
            "validation_size": validation_size,
            "drop_low_variance": drop_low_variance,
            "random_state": RANDOM_STATE,
            "generated_files": [
                validation_path.name,
                test_path.name,
                low_variance_path.name,
                hyperparameters_path.name,
                search_results_path.name,
                "run_config.json",
            ],
        },
    )

    latest_dir = update_latest_files(
        results_root=results_root,
        run_type="baseline",
        source_paths=[
            validation_path,
            test_path,
            low_variance_path,
            hyperparameters_path,
            search_results_path,
            config_path,
        ],
    )
    print(f"\nRun ID: {run_id}")
    print(f"Run directory: {run_dir}")
    print(f"Latest baseline copy: {latest_dir}")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "validation_path": str(validation_path),
        "test_path": str(test_path),
        "low_var_path": str(low_variance_path),
        "hparams_path": str(hyperparameters_path),
        "search_results_path": str(search_results_path),
        "config_path": str(config_path),
    }


def main() -> None:
    """Ejecuta la configuración final del paso 1: entrenamiento y evaluación de modelos tradicionales."""
    # Configuración usada en este experimento.
    subset = "FD004"
    validation_size = DEFAULT_VALIDATION_SIZE
    drop_low_variance = True
    run_id = "FD004_baseline"
    outputs = run_paso_01_pipeline(
        subset=subset,
        validation_size=validation_size,
        drop_low_variance=drop_low_variance,
        run_id=run_id,
    )
    print("\n[PASO 01] Ejecutada correctamente")
    print(f"[PASO 01] run_id: {outputs['run_id']}")
    print(f"[PASO 01] run_dir: {outputs['run_dir']}")


if __name__ == "__main__":
    main()

