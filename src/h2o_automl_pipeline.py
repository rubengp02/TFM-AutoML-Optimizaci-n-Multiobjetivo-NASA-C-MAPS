"""Utilidades de H2O AutoML para regresión de RUL en C-MAPSS."""

# IMPORTS

from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import DEFAULT_H2O_NFOLDS

# IMPORTACIÓN Y COMPROBACIÓN DE H2O

try:
    import h2o
    from h2o.automl import H2OAutoML
except ImportError as exc:  # Si no se puede importar H2O, se asigna None a las herramientas importadas.
    h2o = None
    H2OAutoML = None
    _H2O_IMPORT_ERROR = exc
else:
    _H2O_IMPORT_ERROR = None


def ensure_h2o_installed() -> None:
    """Lanza un error claro si H2O no está instalado."""


    if _H2O_IMPORT_ERROR is not None:
        raise ImportError("H2O is not installed. Install it with: pip install h2o") from _H2O_IMPORT_ERROR


def initialize_h2o(max_mem_size: str | None = None, nthreads: int = -1) -> None:
    """Inicializa H2O de forma segura y evita problemas de arranque repetidos."""


    ensure_h2o_installed() # Se comprueba que H2O está instalado.
    if h2o.cluster() is None:
        h2o.init(max_mem_size=max_mem_size, nthreads=nthreads)
    else:
        h2o.no_progress()


# PREPARACIÓN DE DATOS PARA H2O

# H2O tiene su propia clase H2OFrame, por lo que se necesita una función que convierta DataFrames de pandas a H2OFrames.
def pandas_to_h2o_frame(dataframe: pd.DataFrame, frame_name: str | None = None):
    """Convierte un DataFrame de pandas en un H2OFrame."""
    return h2o.H2OFrame(dataframe, destination_frame=frame_name)


def get_h2o_feature_columns(train_df: pd.DataFrame) -> list[str]:
    """Devuelve las columnas predictoras usadas por H2O AutoML.

    Se excluyen:
    - unit: identificador del motor, no debe usarse como variable predictora.
    - RUL: variable objetivo que el modelo debe predecir.
    """


    columns_to_exclude = {"unit", "RUL"}

    feature_columns = []
    for column_name in train_df.columns:
        if column_name not in columns_to_exclude:
            feature_columns.append(column_name) # Se guardan en una lista las columnas predictoras a usar en H2O AutoML.

    return feature_columns


# CONFIGURACIÓN DE LA EJECUCIÓN AUTOML

def build_default_project_name(subset: str, max_models: int, seed: int, suffix: str = "final") -> str:
    """Construye un nombre descriptivo para una ejecución de H2O AutoML."""

    subset_norm = subset.upper() # Nombre del subset en mayúsculas.

    project_name_parts = [
        f"cmapss_{subset_norm}",
        "h2o",
        f"maxmodels_{max_models}",
        f"seed_{seed}",
        suffix,
    ]

    project_name = "_".join(project_name_parts) # Se unen todas las partes.

    return project_name


# ENTRENAMIENTO CON H2O AUTOML

# Función principal para entrenar H2O AutoML dado un subset, con opciones de configuración y exportación de resultados mediante el uso de las funciones anteriores.
def train_h2o_automl_regressor(train_df: pd.DataFrame, subset: str, max_models: int = 10, max_runtime_secs: int | None = None, seed: int = 42) -> dict[str, Any]:
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
    train_h2o = pandas_to_h2o_frame(train_df, frame_name=f"train_{subset_norm}")

    # Se configura y entrena el modelo H2O AutoML.
    automl_model = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        nfolds=DEFAULT_H2O_NFOLDS,
        seed=seed,
        keep_cross_validation_predictions=True, # Necesario para que H2O pueda construir Stacked Ensembles.
        sort_metric="RMSE",
        project_name=run_project_name,
    )

    # Se mide el tiempo de entrenamiento.
    start = perf_counter()
    automl_model.train(x=features, y="RUL", training_frame=train_h2o)
    train_time_s = perf_counter() - start

    # Se obtiene el leaderboard como DataFrame de pandas para su posterior análisis.
    leaderboard_df = automl_model.leaderboard.as_data_frame()

    return {
        "subset": subset_norm,
        "run_project_name": run_project_name,
        "features": features,
        "leader": automl_model.leader, # El modelo líder entrenado por H2O AutoML.
        "leaderboard": leaderboard_df,
        "train_time_s": float(train_time_s),
        "automl": automl_model,
    }


# EVALUACIÓN DEL MODELO LÍDER EN TEST OFICIAL

def evaluate_h2o_leader_on_test(leader, test_df: pd.DataFrame, y_test: pd.Series, feature_cols: list[str]) -> dict[str, Any]:
    """Ejecuta inferencia del líder y calcula métricas de sklearn en el test oficial."""

    ensure_h2o_installed() # Comprobación de que H2O está instalado.

    evaluation_df = test_df.copy()
    evaluation_df["RUL"] = y_test.reset_index(drop=True) # Se añade RUL al DataFrame.

    test_h2o = pandas_to_h2o_frame(evaluation_df, frame_name="test_official_eval") # Se convierte el DataFrame de test a H2OFrame.

    inference_start_time = perf_counter() # Se mide el tiempo de inferencia.
    h2o_predictions = leader.predict(test_h2o) # Se ejecuta la predicción.
    infer_time_s = perf_counter() - inference_start_time

    predictions_df = h2o_predictions.as_data_frame()
    y_pred = predictions_df.iloc[:, 0].to_numpy(dtype=float)

    y_true = y_test.to_numpy(dtype=float)

    # Cálculo de métricas homogéneo con los modelos tradicionales.
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "inference_time_s": float(infer_time_s),
        "inference_time_per_sample_s": float(infer_time_s / len(evaluation_df)),
        "n_features": int(len(feature_cols)),
        "predictions": y_pred,
    }


# EXPORTACIÓN DE HIPERPARÁMETROS

def export_h2o_leader_hyperparameters(leader, subset: str) -> pd.DataFrame:
    """Extrae los hiperparámetros del mejor modelo de H2O AutoML."""

    subset_norm = subset.upper()
    model_id = str(leader.model_id) # Se obtiene el ID del modelo líder. Atributo propio de H2O.

    # Lista donde se guardará una fila por cada hiperparámetro.
    hyperparameter_rows = []

    # Se recorre el diccionario de hiperparámetros del modelo líder. Guardando: 
    # - Nombre del hiperparámetro 
    # - ID del modelo
    # - Valor actual del hiperparámetro.
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
    model_ids = leaderboard_df["model_id"].astype(str).tolist()

    # Recorremos cada modelo del leaderboard, que ya está ordenado por RMSE de mejor a peor, y extraemos sus hiperparámetros.
    for model_id in model_ids:
        model = h2o.get_model(model_id) # Se obtiene el modelo usado por su ID. 

        # H2O guarda los parámetros de cada modelo en model.params.
        for parameter_name, parameter_info in model.params.items():
            hyperparameter_rows.append(
                {
                    "subset": subset_norm,
                    "model_id": model_id,
                    "parameter": str(parameter_name),
                    "value": str(parameter_info.get("actual")),
                }
            )

    # De esta forma se obtiene un DataFrame con una fila por cada hiperparámetro de cada modelo. Manteniendo compatibilidad con cada modelo, pues cada uno puede tener un conjunto 
    # diferente de hiperparámetros.
    hyperparameters_df = pd.DataFrame(hyperparameter_rows, columns=["subset", "model_id", "parameter", "value"])

    return hyperparameters_df
