"""Modelos de regresión baseline para la predicción de RUL en C-MAPSS."""

# IMPORTS
from __future__ import annotations

import warnings
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

from src.config import LOW_VARIANCE_THRESHOLD


# CONSTRUCCIÓN DE MODELOS TRADICIONALES.


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
        n_jobs=-1,
    )


def build_extra_trees(random_state: int) -> ExtraTreesRegressor:
    """Construye ExtraTrees como baseline de árboles altamente aleatorizados."""

    return ExtraTreesRegressor(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
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
        n_jobs=-1,
    )


def build_lightgbm(random_state: int):
    """Construye LightGBM si la librería está disponible."""

    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
    )


def build_baseline_models(random_state: int = 42) -> dict[str, object]:
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


# PREPARACIÓN DE VARIABLES Y VALIDACIÓN


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
    numeric_variances = train_df[base_features].select_dtypes(include=[np.number]).var(numeric_only=True)

    low_variance_columns = numeric_variances[numeric_variances <= low_var_threshold].index.tolist()

    selected_features = []
    for column_name in base_features:
        if column_name not in low_variance_columns:
            selected_features.append(column_name)

    return selected_features, sorted(low_variance_columns)



def split_train_validation_by_unit(train_df: pd.DataFrame, validation_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide train según motores completos (unit), no por filas, ya que podríamos acabar con ciclos del mismo motor en entrenamiento 
    y en validación si no se hiciera.

    Esto evita mezclar ciclos del mismo motor entre train y validación.
    """

    unit_ids = train_df["unit"].drop_duplicates().values # Se eliminan duplicados, quedando los IDs únicos de motores.
    train_units, val_units = train_test_split( 
        unit_ids,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    ) # Dividimos los IDs aleatoriamente en conjunto de entrenamiento y validación.

    # Se seleccionan las filas correspondientes a los IDs de entrenamiento y validación respectivamente, asegurando que cada motor completo ("unit") esté solo en uno de los conjuntos.
    train_split = train_df[train_df["unit"].isin(train_units)].copy() 
    val_split = train_df[train_df["unit"].isin(val_units)].copy()
    return train_split, val_split


# ENTRENAMIENTO Y EVALUACIÓN DE MODELOS


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
    infer_start = perf_counter()
    y_pred = model.predict(X_eval)
    inference_time = perf_counter() - infer_start


    return y_pred, train_time, inference_time 




def evaluate_single_model(model_name: str, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series, feature_cols: list[str], 
                          removed_low_variance_cols: list[str]) -> dict[str, float | str | int]:
    """Entrena un modelo, mide tiempos y calcula métricas en el conjunto de evaluación."""

    y_pred, train_time, infer_time = _fit_predict_timed(model, X_train, y_train, X_eval) # Obtenemos predicciones y tiempos de entrenamiento e inferencia.
    metrics = _evaluate_predictions(y_eval, y_pred) # Se calculan las métricas de evaluación.
    removed_columns_text = ";".join(removed_low_variance_cols) # Se crea un string con las columnas eliminadas, separadas por ";", para que sea fácil almacenar en el CSV de resultados.

    return {
        "model": model_name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "train_time_s": float(train_time),
        "inference_time_s": float(infer_time),
        "inference_time_per_sample_s": float(infer_time / len(X_eval)),
        "n_features": int(len(feature_cols)),
        "removed_low_variance_columns": removed_columns_text,
    } # Se devuelve un diccionario con el nombre del modelo, las métricas, tiempos y número de características usadas, así como las columnas eliminadas por baja varianza (si las hay).



def evaluate_models(models: dict[str, object], train_df: pd.DataFrame, eval_df: pd.DataFrame, eval_target: pd.Series, 
                    feature_cols: list[str], removed_low_variance_cols: list[str]) -> pd.DataFrame:
    """Entrena todos los modelos con train_df y los evalúa sobre eval_df.

    El conjunto de evaluación puede ser validación interna o test oficial.
    No se utiliza para ajustar los modelos.
    """

    # Se separan las características y target en train y validation.
    X_train = train_df[feature_cols]
    y_train = train_df["RUL"]
    X_eval = eval_df[feature_cols]

    # Se crea una lista de diccionarios, donde se guarda una fila de resultados por cada modelo evaluado.
    result_rows = []
    for model_name, model in models.items(): # Se itera sobre cada modelo.
        result_rows.append(
            evaluate_single_model(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_eval=X_eval,
                y_eval=eval_target,
                feature_cols=feature_cols,
                removed_low_variance_cols=removed_low_variance_cols,
            )
        )

    # Se devuelve un DataFrame con los resultados de todos los modelos, ordenado por RMSE de mejor a peor.
    return pd.DataFrame(result_rows).sort_values("RMSE", ascending=True).reset_index(drop=True) 


# HIPERPARÁMETROS


def collect_model_hyperparameters(models: dict[str, object], subset: str) -> pd.DataFrame: 
    """Extrae hiperparámetros con get_params() en formato largo y se guarda en un DataFrame para posterior análisis."""

    subset_norm = subset.upper()
    hyperparameter_rows = [] # Lista vacía donde se guardará la información de hiperparámetros.

    # Se itera sobre cada modelo del diccionario.
    for model_name, model in models.items():
        model_parameters = model.get_params(deep=True) # Se obtienen los hiperparámetros de cada modelo.

        # Se itera sobre los hiperparámetros y sus valores.
        for parameter_name, value in model_parameters.items():
            hyperparameter_rows.append(
                {
                    "subset": str(subset_norm),
                    "model": str(model_name),
                    "parameter": str(parameter_name),
                    "value": str(value),
                }
            ) # Se guardan los hiperparámetros en un diccionario, en str para evitar problemas en CSV.

    # Se devuelve un dataframe con los hiperparámetros de todos los modelos y con las siguientes columnas: subset, model, parameter, value.
    return pd.DataFrame(hyperparameter_rows, columns=["subset", "model", "parameter", "value"])


# GUARDADO DE RESULTADOS


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
