"""Optimización multiobjetivo NSGA-II para modelos de compromiso seleccionados de C-MAPSS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.traditional_models import get_feature_columns, split_train_validation_by_unit
from src.config import (
    DEFAULT_MAX_RMSE_DEGRADATION,
    DEFAULT_MULTI_OBJECTIVES,
    SUPPORTED_NSGA2_MODELS,
)
from src.pareto_compromise_selection import select_best_compromise_solution
from src.cmapss_data_preparation import prepare_cmapss_data
from src.model_comparison import detect_pareto_solutions

try:
    import optuna
    from optuna.samplers import NSGAIISampler
except ImportError as exc:  # pragma: no cover
    optuna = None
    NSGAIISampler = None
    _OPTUNA_IMPORT_ERROR = exc
else:
    _OPTUNA_IMPORT_ERROR = None


SUPPORTED_MODEL_SHORT = SUPPORTED_NSGA2_MODELS


@dataclass
class Nsga2Artifacts:
    """Contenedor para las salidas de la optimización NSGA-II."""

    selected_model_short: str
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
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise ImportError("XGBoost is required for selected model. Install with: pip install xgboost") from exc

    if model_short == "LightGBM":
        try:
            import lightgbm  # noqa: F401
        except ImportError as exc:
            raise ImportError("LightGBM is required for selected model. Install with: pip install lightgbm") from exc


def _sample_hyperparameters(trial, model_short: str) -> dict[str, Any]:
    """Define el espacio de búsqueda según el modelo seleccionado."""
    if model_short == "Ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        }

    if model_short == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        }

    if model_short == "ExtraTrees":
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
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=float(sampled_hyperparameters["alpha"]))),
            ]
        )

    if model_short == "RandomForest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **sampled_hyperparameters)

    if model_short == "ExtraTrees":
        return ExtraTreesRegressor(random_state=random_state, n_jobs=-1, **sampled_hyperparameters)

    if model_short == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=random_state, **sampled_hyperparameters)

    if model_short == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            **sampled_hyperparameters,
        )

    if model_short == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=random_state, n_jobs=-1, **sampled_hyperparameters)

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

    from src.pareto_compromise_selection import get_best_compromise_row

    best_row, candidates_df = get_best_compromise_row(
        comparison_df=comparison_df,
        max_rmse_degradation=max_rmse_degradation,
    )
    model_short = str(best_row.get("model_short", ""))

    if model_short not in SUPPORTED_MODEL_SHORT:
        raise ValueError(f"Model '{model_short}' is not supported for NSGA-II optimization.")

    _require_model_dependency(model_short)

    data_bundle = prepare_cmapss_data(subset=subset, raw_dir=raw_dir)
    train_df = data_bundle["train_df"]
    test_last_df = data_bundle["test_last_df"]
    y_test = data_bundle["y_test"]

    # Methodological note: split by full engine IDs to avoid leaking temporal
    # trajectories from one unit between train and validation.
    train_split_df, val_split_df = split_train_validation_by_unit(
        train_df,
        validation_size=validation_size,
        random_state=random_state,
    )

    feature_cols_val, removed_low_var_val = get_feature_columns(train_split_df, drop_low_variance=True)
    X_train_val = train_split_df[feature_cols_val]
    y_train_val = train_split_df["RUL"]
    X_val = val_split_df[feature_cols_val]
    y_val = val_split_df["RUL"]

    sampler = NSGAIISampler(seed=random_state, population_size=population_size)
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"], sampler=sampler)

    def objective(trial):
        sampled_hyperparameters = _sample_hyperparameters(trial, model_short=model_short)
        trial_model = _build_model(model_short, sampled_hyperparameters=sampled_hyperparameters, random_state=random_state)

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

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    completed_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not completed_trials:
        raise ValueError("NSGA-II did not produce completed trials.")

    pareto_trial_numbers = {trial.number for trial in study.best_trials}

    validation_rows: list[dict[str, Any]] = []
    for trial in completed_trials:
        sampled_hyperparameters = trial.user_attrs.get("params", {})
        trial_model = _build_model(model_short, sampled_hyperparameters=sampled_hyperparameters, random_state=random_state)
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
        trial_model = _build_model(model_short, sampled_hyperparameters=sampled_hyperparameters, random_state=random_state)

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
                "inference_time_s": inference_time_s_test,
                "inference_time_per_sample_s": float(inference_time_s_test / len(X_test)),
                "n_features": int(len(feature_cols_test)),
                "removed_low_variance_columns": ";".join(removed_low_var_test),
                "params_json": json.dumps(sampled_hyperparameters, sort_keys=True),
            }
        )

    if not test_rows:
        raise ValueError("No Pareto-optimal validation trials available for official test evaluation.")

    test_results_df = pd.DataFrame(test_rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    pareto_test_df = detect_pareto_solutions(
        test_results_df,
        objectives=DEFAULT_MULTI_OBJECTIVES,
    )
    pareto_test_df = pareto_test_df[pareto_test_df["is_pareto_optimal"]].copy().reset_index(drop=True)

    best_compromise_test_df = select_best_compromise_solution(
        df=test_results_df,
        objective_cols=DEFAULT_MULTI_OBJECTIVES,
        time_cols=["train_time_s", "inference_time_s"],
        weights=None,
    ).sort_values("ideal_distance", ascending=True).head(1)

    return Nsga2Artifacts(
        selected_model_short=model_short,
        compromise_selection_df=candidates_df,
        trials_validation_df=trials_validation_df,
        pareto_validation_df=pareto_validation_df,
        test_results_df=test_results_df,
        pareto_test_df=pareto_test_df,
        best_compromise_test_df=best_compromise_test_df,
    )
