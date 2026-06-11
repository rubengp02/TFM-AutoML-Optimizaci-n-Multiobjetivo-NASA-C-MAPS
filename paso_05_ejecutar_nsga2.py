"""PASO 05: optimización multiobjetivo NSGA-II."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_MAX_RMSE_DEGRADATION, DEFAULT_MULTI_OBJECTIVES, DEFAULT_NSGA2_POPULATION_SIZE, DEFAULT_NSGA2_TRIALS, DEFAULT_VALIDATION_SIZE, ENSEMBLE_TOKENS, RANDOM_STATE, SUPPORTED_NSGA2_MODELS
from src.cmapss_data_preparation import get_project_root, prepare_cmapss_data
from src.experiment_tracking import clean_directory_contents_for_stable_run, create_run_directory, generate_run_id, save_run_config

STABLE_RUN_IDS = {"FD004_baseline", "FD004_h2o_30models", "FD004_comparison_initial", "FD004_compromise_selection", "FD004_nsga2_60trials", "FD004_comparison_final"}

try:
    import optuna
    from optuna.samplers import NSGAIISampler
except ImportError as exc:  # pragma: no cover
    optuna = None
    NSGAIISampler = None
    _OPTUNA_IMPORT_ERROR = exc
else:
    _OPTUNA_IMPORT_ERROR = None


# UTILIDADES DE FEATURES Y PARETO-COMPROMISO

def get_feature_columns(train_df: pd.DataFrame, drop_low_variance: bool = True, low_var_threshold: float = 1e-8) -> tuple[list[str], list[str]]:
    """Devuelve las columnas de entrada y las columnas eliminadas por baja varianza.
    
    La varianza baja se calcula solo en entrenamiento para evitar fuga de información.
    """
    columns_to_exclude = {"RUL", "unit"}
    base_features = []
    for column_name in train_df.columns:
        if column_name not in columns_to_exclude:
            base_features.append(column_name)
    if not drop_low_variance:
        return base_features, []

    numeric_variances = train_df[base_features].select_dtypes(
        include=[np.number],
    ).var(numeric_only=True)
    low_variance_columns = numeric_variances[
        numeric_variances <= low_var_threshold
    ].index.tolist()
    selected_features = []
    for column_name in base_features:
        if column_name not in low_variance_columns:
            selected_features.append(column_name)
    return selected_features, sorted(low_variance_columns)


def split_train_validation_by_unit(train_df: pd.DataFrame, validation_size: float = DEFAULT_VALIDATION_SIZE, random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide train según motores completos (unit), no por filas, ya que podríamos acabar con ciclos del mismo motor en entrenamiento 
    y en validación si no se hiciera.
    
    Esto evita mezclar ciclos del mismo motor entre train y validación.
    """
    # Methodological note: split by full engine IDs to avoid leaking temporal
    # trajectories from one unit between train and validation.
    unit_ids = train_df["unit"].drop_duplicates().values
    train_units, validation_units = train_test_split(
        unit_ids,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )
    train_split_df = train_df[train_df["unit"].isin(train_units)].copy()
    validation_split_df = train_df[train_df["unit"].isin(validation_units)].copy()
    return train_split_df, validation_split_df


def detect_pareto_solutions(df: pd.DataFrame, objectives: list[str]) -> pd.DataFrame:
    """Marca filas no dominadas (Pareto-óptimas) para objetivos de minimización."""
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


def _safe_minmax(series: pd.Series) -> pd.Series:
    """Aplica normalización min-max con tratamiento seguro de valores constantes."""
    minimum_value = float(series.min())
    maximum_value = float(series.max())
    if maximum_value - minimum_value <= 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - minimum_value) / (maximum_value - minimum_value)


def filter_supported_non_ensemble_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva filas Pareto-óptimas compatibles y que no sean ensemble/stacked."""
    model_column = "model_short" if "model_short" in df.columns else "model"
    pareto_candidates_df = df[df["is_pareto_optimal"] == True].copy()  # noqa: E712
    model_text = pareto_candidates_df[model_column].astype(str).str.lower()
    is_ensemble = model_text.apply(lambda name: any(token in name for token in ENSEMBLE_TOKENS))
    is_supported = pareto_candidates_df[model_column].astype(str).isin(SUPPORTED_NSGA2_MODELS)
    return pareto_candidates_df[~is_ensemble & is_supported].copy().reset_index(drop=True)


def select_best_compromise_solution(df: pd.DataFrame, objective_cols: Iterable[str], time_cols: Iterable[str], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Selecciona el mejor compromiso por distancia normalizada al punto ideal.
    
    Pasos:
    - Usa las columnas objetivo proporcionadas.
    - Aplica log1p a objetivos de tiempo antes de normalizar.
    - Aplica normalización min-max.
    - Calcula distancia euclídea ponderada al punto ideal (todo ceros).
    """
    objectives = list(objective_cols)
    time_set = set(time_cols)
    scored_df = df.copy().reset_index(drop=True)
    # Build normalized columns required by the user.
    norm_column_map = {"RMSE": "RMSE_norm", "train_time_s": "train_time_s_norm", "inference_time_s": "inference_time_s_norm"}
    for objective_name in objectives:
        transformed = np.log1p(scored_df[objective_name].astype(float)) if objective_name in time_set else scored_df[objective_name].astype(float)
        scored_df[norm_column_map.get(objective_name, f"{objective_name}_norm")] = _safe_minmax(transformed)
    # Default equal weights on normalized objectives.
    objective_weights = {objective_name: float(weights.get(objective_name, 1.0)) for objective_name in objectives} if weights else {objective_name: 1.0 for objective_name in objectives}
    distance_squared = np.zeros(len(scored_df), dtype=float)
    for objective_name in objectives:
        normalized_column = norm_column_map.get(objective_name, f"{objective_name}_norm")
        distance_squared += objective_weights[objective_name] * np.square(scored_df[normalized_column].astype(float).to_numpy())
    # Weighted Euclidean distance to ideal point (0,0,...).
    scored_df["ideal_distance"] = np.sqrt(distance_squared)
    best_index = int(scored_df["ideal_distance"].idxmin())
    scored_df["is_best_compromise"] = False
    scored_df.loc[best_index, "is_best_compromise"] = True
    return scored_df


def get_best_compromise_row(comparison_df: pd.DataFrame, max_rmse_degradation: float = DEFAULT_MAX_RMSE_DEGRADATION) -> tuple[pd.Series, pd.DataFrame]:
    """Devuelve la mejor fila de compromiso y la tabla ordenada de candidatos para configurar NSGA-II."""
    candidates_df = filter_supported_non_ensemble_pareto(comparison_df)
    best_rmse = float(candidates_df["RMSE"].min())
    rmse_threshold = best_rmse * (1.0 + float(max_rmse_degradation))
    candidates_df = candidates_df.copy()
    candidates_df["passes_rmse_filter"] = candidates_df["RMSE"] <= rmse_threshold
    # Filtro de calidad: evita seleccionar modelos excesivamente rápidos pero imprecisos.
    filtered_candidates_df = candidates_df[candidates_df["passes_rmse_filter"]].copy().reset_index(drop=True)
    scored_df = select_best_compromise_solution(
        filtered_candidates_df,
        objective_cols=["RMSE", "train_time_s", "inference_time_s"],
        time_cols=["train_time_s", "inference_time_s"],
        weights=None,
    )
    scored_df = scored_df.sort_values("ideal_distance", ascending=True).reset_index(drop=True)
    return scored_df.iloc[0], scored_df


# NSGA-II

@dataclass
class Nsga2Artifacts:
    selected_model_short: str
    search_time_s: float
    compromise_selection_df: pd.DataFrame
    trials_validation_df: pd.DataFrame
    pareto_validation_df: pd.DataFrame
    test_results_df: pd.DataFrame
    pareto_test_df: pd.DataFrame
    best_compromise_test_df: pd.DataFrame


def ensure_optuna_installed() -> None:
    """Lanza un error claro cuando Optuna no está disponible."""
    if _OPTUNA_IMPORT_ERROR is not None:
        raise ImportError("Optuna is not installed. Install it with: pip install optuna") from _OPTUNA_IMPORT_ERROR


def _require_model_dependency(model_short: str) -> None:
    """Valida dependencias opcionales de modelos antes de optimizar."""
    if model_short == "XGBoost":
        import xgboost  # noqa: F401
    if model_short == "LightGBM":
        import lightgbm  # noqa: F401


def _sample_hyperparameters(trial, model_short: str) -> dict[str, Any]:
    """Define el espacio de búsqueda según el modelo seleccionado."""
    if model_short == "Ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}
    if model_short in {"RandomForest", "ExtraTrees"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        }
    if model_short == "HistGradientBoosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "max_iter": trial.suggest_int("max_iter", 120, 700),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 120),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 1.0, log=True),
        }
    if model_short == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if model_short == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    raise ValueError(f"Unsupported model_short for NSGA-II: {model_short}")


def _build_model(model_short: str, sampled_hyperparameters: dict[str, Any], random_state: int):
    """Instancia un modelo a partir de model_short y parámetros muestreados."""
    if model_short == "Ridge":
        return Pipeline(steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=float(sampled_hyperparameters["alpha"])))])
    if model_short == "RandomForest":
        return RandomForestRegressor(random_state=random_state, n_jobs=4, **sampled_hyperparameters)
    if model_short == "ExtraTrees":
        return ExtraTreesRegressor(random_state=random_state, n_jobs=4, **sampled_hyperparameters)
    if model_short == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=random_state, **sampled_hyperparameters)
    if model_short == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:squarederror", random_state=random_state, n_jobs=4, **sampled_hyperparameters)
    if model_short == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=random_state, n_jobs=4, **sampled_hyperparameters)
    raise ValueError(f"Unsupported model_short for NSGA-II: {model_short}")


def _evaluate_model(model, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series) -> tuple[float, float, float, float, float]:
    """Ajusta y evalúa el modelo devolviendo MAE, RMSE, R2, train_time_s e inference_time_s."""
    t0 = perf_counter()
    model.fit(X_train, y_train)
    train_time_s = perf_counter() - t0
    t1 = perf_counter()
    y_pred = model.predict(X_eval)
    inference_time_s = perf_counter() - t1
    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    r2 = float(r2_score(y_eval, y_pred))
    return mae, rmse, r2, float(train_time_s), float(inference_time_s)


def optimize_with_nsga2(subset: str, comparison_df: pd.DataFrame, n_trials: int, population_size: int, validation_size: float, random_state: int, max_rmse_degradation: float = DEFAULT_MAX_RMSE_DEGRADATION, raw_dir: Path | None = None) -> Nsga2Artifacts:
    """Ejecuta optimización NSGA-II para el modelo de compromiso seleccionado automáticamente."""
    ensure_optuna_installed()
    best_row, candidates_df = get_best_compromise_row(comparison_df=comparison_df, max_rmse_degradation=max_rmse_degradation)
    model_short = str(best_row.get("model_short", ""))
    if model_short not in SUPPORTED_NSGA2_MODELS:
        raise ValueError(f"Model '{model_short}' is not supported for NSGA-II optimization.")
    _require_model_dependency(model_short)

    data_bundle = prepare_cmapss_data(subset=subset, raw_dir=raw_dir)
    train_df = data_bundle["train_df"]
    test_last_df = data_bundle["test_last_df"]
    y_test = data_bundle["y_test"]

    train_split_df, validation_split_df = split_train_validation_by_unit(train_df, validation_size=validation_size, random_state=random_state)
    feature_cols_val, removed_low_var_val = get_feature_columns(train_split_df, drop_low_variance=True)
    X_train_val = train_split_df[feature_cols_val]
    y_train_val = train_split_df["RUL"]
    X_val = validation_split_df[feature_cols_val]
    y_val = validation_split_df["RUL"]

    sampler = NSGAIISampler(seed=random_state, population_size=population_size)
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"], sampler=sampler)

    def objective(trial):
        sampled_hyperparameters = _sample_hyperparameters(trial, model_short=model_short)
        trial_model = _build_model(
            model_short,
            sampled_hyperparameters=sampled_hyperparameters,
            random_state=random_state,
        )
        _, rmse, _, train_time_s, inference_time_s = _evaluate_model(
            trial_model,
            X_train=X_train_val,
            y_train=y_train_val,
            X_eval=X_val,
            y_eval=y_val,
        )
        trial.set_user_attr("model_short", model_short)
        trial.set_user_attr("params", sampled_hyperparameters)
        trial.set_user_attr("n_features", len(feature_cols_val))
        trial.set_user_attr("removed_low_variance_columns", removed_low_var_val)
        return rmse, train_time_s, inference_time_s

    search_start_time = perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    search_time_s = perf_counter() - search_start_time
    completed_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    pareto_trial_numbers = {trial.number for trial in study.best_trials}

    validation_rows: list[dict[str, Any]] = []
    for trial in completed_trials:
        sampled_hyperparameters = trial.user_attrs.get("params", {})
        trial_model = _build_model(
            model_short,
            sampled_hyperparameters=sampled_hyperparameters,
            random_state=random_state,
        )
        mae_val, rmse_val, r2_val, train_time_val, inference_time_val = _evaluate_model(
            trial_model,
            X_train=X_train_val,
            y_train=y_train_val,
            X_eval=X_val,
            y_eval=y_val,
        )
        validation_rows.append(
            {
                "trial_number": trial.number,
                "model_short": model_short,
                "MAE": mae_val,
                "RMSE": rmse_val,
                "R2": r2_val,
                "train_time_s": train_time_val,
                "search_time_s": float(search_time_s),
                "inference_time_s": inference_time_val,
                "n_features": int(trial.user_attrs.get("n_features", len(feature_cols_val))),
                "removed_low_variance_columns": ";".join(trial.user_attrs.get("removed_low_variance_columns", [])),
                "params_json": json.dumps(sampled_hyperparameters, sort_keys=True),
                "is_pareto_optimal": trial.number in pareto_trial_numbers,
            }
        )

    trials_validation_df = pd.DataFrame(validation_rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    pareto_validation_df = trials_validation_df[trials_validation_df["is_pareto_optimal"]].copy().reset_index(drop=True)

    # Methodological note: the official test set is only used for final reporting.
    # Hyperparameter search decisions are made exclusively on internal validation.
    # We therefore evaluate only non-dominated validation solutions on test.
    feature_cols_test, removed_low_var_test = get_feature_columns(train_df, drop_low_variance=True)
    X_train_full = train_df[feature_cols_test]
    y_train_full = train_df["RUL"]
    X_test = test_last_df[feature_cols_test]
    pareto_validation_trial_numbers = set(pareto_validation_df["trial_number"].astype(int).tolist())

    test_rows: list[dict[str, Any]] = []
    for trial in completed_trials:
        if int(trial.number) not in pareto_validation_trial_numbers:
            continue
        sampled_hyperparameters = trial.user_attrs.get("params", {})
        trial_model = _build_model(
            model_short,
            sampled_hyperparameters=sampled_hyperparameters,
            random_state=random_state,
        )
        mae_test, rmse_test, r2_test, train_time_s_test, inference_time_s_test = _evaluate_model(
            trial_model,
            X_train=X_train_full,
            y_train=y_train_full,
            X_eval=X_test,
            y_eval=y_test,
        )
        test_rows.append(
            {
                "trial_number": trial.number,
                "model_short": model_short,
                "MAE": mae_test,
                "RMSE": rmse_test,
                "R2": r2_test,
                "train_time_s": train_time_s_test,
                "search_time_s": float(search_time_s),
                "inference_time_s": inference_time_s_test,
                "inference_time_per_sample_s": float(inference_time_s_test / len(X_test)),
                "n_features": int(len(feature_cols_test)),
                "removed_low_variance_columns": ";".join(removed_low_var_test),
                "params_json": json.dumps(sampled_hyperparameters, sort_keys=True),
            }
        )

    test_results_df = pd.DataFrame(test_rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    pareto_test_df = detect_pareto_solutions(test_results_df, objectives=DEFAULT_MULTI_OBJECTIVES)
    pareto_test_df = pareto_test_df[pareto_test_df["is_pareto_optimal"]].copy().reset_index(drop=True)
    best_compromise_test_df = select_best_compromise_solution(
        df=test_results_df,
        objective_cols=DEFAULT_MULTI_OBJECTIVES,
        time_cols=["train_time_s", "inference_time_s"],
        weights=None,
    ).sort_values("ideal_distance", ascending=True).head(1)

    return Nsga2Artifacts(
        selected_model_short=model_short,
        search_time_s=float(search_time_s),
        compromise_selection_df=candidates_df,
        trials_validation_df=trials_validation_df,
        pareto_validation_df=pareto_validation_df,
        test_results_df=test_results_df,
        pareto_test_df=pareto_test_df,
        best_compromise_test_df=best_compromise_test_df,
    )


# ORQUESTACIÓN PASO 05

def run_paso_05_pipeline(subset: str, comparison_run_id: str, n_trials: int = DEFAULT_NSGA2_TRIALS, population_size: int = DEFAULT_NSGA2_POPULATION_SIZE, validation_size: float = DEFAULT_VALIDATION_SIZE, max_rmse_degradation: float = DEFAULT_MAX_RMSE_DEGRADATION, random_state: int = RANDOM_STATE, run_id: str | None = None, output_dir: str = "results") -> dict[str, str]:
    """Ejecuta NSGA-II y guarda los artefactos de validación y test."""
    subset = subset.upper()
    project_root = get_project_root()
    results_root = Path(project_root) / output_dir
    comparison_path = results_root / "runs" / comparison_run_id / "comparison.csv"
    comparison_df = pd.read_csv(comparison_path)
    run_id = run_id or generate_run_id(subset=subset, execution_type="nsga2", main_descriptor=f"{n_trials}trials")
    run_dir = create_run_directory(results_root=results_root, run_id=run_id)
    clean_directory_contents_for_stable_run(run_dir, stable_run_ids=STABLE_RUN_IDS)

    artifacts = optimize_with_nsga2(
        subset=subset,
        comparison_df=comparison_df,
        n_trials=n_trials,
        population_size=population_size,
        validation_size=validation_size,
        random_state=random_state,
        max_rmse_degradation=max_rmse_degradation,
        raw_dir=project_root / "data" / "raw",
    )

    compromise_path = run_dir / "compromise_selection.csv"
    trials_val_path = run_dir / "nsga2_trials_validation.csv"
    pareto_val_path = run_dir / "nsga2_pareto_validation.csv"
    test_results_path = run_dir / "nsga2_test_results.csv"
    pareto_test_path = run_dir / "nsga2_pareto_test.csv"
    best_compromise_test_path = run_dir / "nsga2_best_compromise_test.csv"
    artifacts.compromise_selection_df.to_csv(compromise_path, index=False)
    artifacts.trials_validation_df.to_csv(trials_val_path, index=False)
    artifacts.pareto_validation_df.to_csv(pareto_val_path, index=False)
    artifacts.test_results_df.to_csv(test_results_path, index=False)
    artifacts.pareto_test_df.to_csv(pareto_test_path, index=False)
    artifacts.best_compromise_test_df.to_csv(best_compromise_test_path, index=False)

    config_path = save_run_config(
        run_dir,
        {
            "run_id": run_id,
            "subset": subset,
            "execution_type": "nsga2",
            "comparison_run_id": comparison_run_id,
            "n_trials": n_trials,
            "population_size": population_size,
            "validation_size": validation_size,
            "max_rmse_degradation": max_rmse_degradation,
            "random_state": random_state,
            "selected_model_short": artifacts.selected_model_short,
            "search_time_s": artifacts.search_time_s,
            "generated_files": [
                compromise_path.name,
                trials_val_path.name,
                pareto_val_path.name,
                test_results_path.name,
                pareto_test_path.name,
                best_compromise_test_path.name,
                "run_config.json",
            ],
        },
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "compromise_path": str(compromise_path),
        "trials_val_path": str(trials_val_path),
        "pareto_val_path": str(pareto_val_path),
        "test_results_path": str(test_results_path),
        "pareto_test_path": str(pareto_test_path),
        "best_compromise_test_path": str(best_compromise_test_path),
        "config_path": str(config_path),
    }


def main() -> None:
    """Parsea argumentos CLI y ejecuta la optimización NSGA-II sobre el modelo de compromiso seleccionado."""
    # Configuración usada en este experimento.
    subset = "FD004"
    comparison_run_id = "FD004_comparison_initial"
    n_trials = 60
    population_size = 12
    validation_size = DEFAULT_VALIDATION_SIZE
    max_rmse_degradation = DEFAULT_MAX_RMSE_DEGRADATION
    random_state = RANDOM_STATE
    run_id = "FD004_nsga2_60trials"
    output_dir = "results"
    outputs = run_paso_05_pipeline(
        subset=subset,
        comparison_run_id=comparison_run_id,
        n_trials=n_trials,
        population_size=population_size,
        validation_size=validation_size,
        max_rmse_degradation=max_rmse_degradation,
        random_state=random_state,
        run_id=run_id,
        output_dir=output_dir,
    )
    print("\n[PASO 05] Ejecutada correctamente")
    print(f"[PASO 05] run_id: {outputs['run_id']}")
    print(f"[PASO 05] run_dir: {outputs['run_dir']}")


if __name__ == "__main__":
    main()

