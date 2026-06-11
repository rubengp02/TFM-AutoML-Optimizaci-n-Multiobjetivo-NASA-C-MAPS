# C-MAPSS Experimental Project (FD001 / FD004)

Pipeline for RUL prediction experiments on NASA C-MAPSS with traditional models and H2O AutoML.

## Run-Scoped Results

Each execution creates an isolated folder:
- `results/runs/{run_id}/...`

A convenience copy of the latest outputs is also updated in:
- `results/latest/baseline/`
- `results/latest/h2o/`
- `results/latest/comparison/`
- `results/latest/comparison/figures/`

## Scripts

- Baselines by subset:
  - `python run_cmapss_baseline.py --subset FD001`
  - `python run_cmapss_baseline.py --subset FD004`
  - Optional: `--run-id`

- H2O AutoML by subset:
  - `python run_h2o_automl.py --subset FD001 --max-models 10`
  - `python run_h2o_automl.py --subset FD004 --max-models 30`
  - Optional: `--run-id`
  - El `project_name` de H2O se genera automáticamente con `build_default_project_name(...)`.
  - Final setup includes `DeepLearning` and `StackedEnsemble` (no default exclusions).
  - `keep_cross_validation_predictions=True` is enabled to support stable stacked ensembles.

- Result comparison and Pareto:
  - `python run_compare_results.py --subset FD004 --baseline-run-id <BASELINE_RUN_ID> --h2o-run-id <H2O_RUN_ID>`
  - Optional: `--run-id`

## Files Per Run

Baseline run folder (`results/runs/{run_id}/`):
- `baseline_validation.csv`
- `baseline_test.csv`
- `low_variance_columns.csv`
- `hyperparameters_baseline.csv`
- `run_config.json`

H2O run folder (`results/runs/{run_id}/`):
- `h2o_leaderboard.csv`
- `h2o_test_results.csv`
- `hyperparameters_h2o_leader.csv`
- `hyperparameters_h2o_all_models.csv`
- `run_config.json`
  - En los CSV de hiperparámetros H2O se guardan columnas: `subset`, `model_id`, `parameter`, `value`.

Comparison run folder (`results/runs/{run_id}/`):
- `comparison.csv`
- `comparison_compact.csv`
- `comparison_latex.csv`
- `pareto.csv`
- `figures/`
- `run_config.json`

## Ejecucion experimental final

```powershell
pip install -r requirements.txt
python run_cmapss_baseline.py --subset FD004
python run_h2o_automl.py --subset FD004 --max-models 30
python run_compare_results.py --subset FD004 --baseline-run-id <BASELINE_RUN_ID> --h2o-run-id <H2O_RUN_ID>
```

## Flujo experimental por pasos (entrypoints oficiales)

```powershell
python paso_01_entrenar_modelos_tradicionales.py --subset FD004
python paso_02_entrenar_h2o_automl.py --subset FD004 --max-models 30
python paso_03_comparacion_inicial.py --subset FD004 --baseline-run-id <BASELINE_RUN_ID> --h2o-run-id <H2O_RUN_ID>
python paso_04_seleccionar_compromiso_pareto.py --subset FD004 --comparison-run-id <COMPARISON_RUN_ID>
python paso_05_ejecutar_nsga2.py --subset FD004 --comparison-run-id <COMPARISON_RUN_ID> --n-trials 60 --population-size 12
python paso_06_comparacion_final.py --subset FD004 --baseline-run-id <BASELINE_RUN_ID> --h2o-run-id <H2O_RUN_ID> --nsga2-run-id <NSGA2_RUN_ID>
python paso_07_generar_figuras_finales.py --results-dir results/final
```

Opcional:

```powershell
python run_full_pipeline.py --subset FD004 --max-models 30 --n-trials 60 --population-size 12
```

## Arquitectura del codigo

### Scripts por PASO (`paso_*.py`)

- `paso_01_entrenar_modelos_tradicionales.py`:
  carga un subset C-MAPSS, ejecuta exploración básica, entrena/evalúa modelos tradicionales y guarda artefactos de la corrida.
- `paso_02_entrenar_h2o_automl.py`:
  entrena H2O AutoML, evalúa el leader en test oficial y exporta leaderboard, resultados y hiperparámetros.
- `paso_03_comparacion_inicial.py`:
  combina resultados de baseline/H2O, recalcula rankings/Pareto y genera tablas y figuras de comparación inicial.
- `paso_04_seleccionar_compromiso_pareto.py`:
  selecciona automáticamente el modelo candidato de compromiso no ensemble para la fase NSGA-II.
- `paso_05_ejecutar_nsga2.py`:
  ejecuta NSGA-II en validación interna por `unit` y evalúa en test solo las soluciones no dominadas de validación.
- `paso_06_comparacion_final.py`:
  combina baseline/H2O/NSGA-II y genera la comparación final con Pareto y figuras.
- `paso_07_generar_figuras_finales.py`:
  genera las figuras finales para memoria a partir de CSV ya calculados en `results/final/`, sin recalcular modelos.

### Modulos principales (`src/*.py`)

- `cmapss_data_preparation.py`: carga robusta de C-MAPSS por subset y preparacion de `RUL`.
- `exploration.py`: resumenes estructurales de datos y diagnosticos de calidad.
- `traditional_models.py`: modelos tradicionales, metrica/tiempos, columnas de features y export de hiperparametros.
- `h2o_automl_pipeline.py`: utilidades H2O AutoML (entrenamiento, evaluacion y exportes).
- `model_comparison.py`: union de resultados, rankings, Pareto y figuras.
- `pareto_compromise_selection.py`: filtro metodologico para seleccionar candidato de compromiso previo a NSGA-II.
- `nsga2_multiobjective_optimization.py`: optimizacion multiobjetivo con Optuna NSGA-II.
- `experiment_tracking.py`: utilidades transversales de `run_id`, carpetas por corrida y `run_config.json`.
- `config.py`: constantes compartidas (seed, objetivos, defaults, etc.).

### Flujo experimental final

1. Baselines en FD004.
2. H2O AutoML en FD004.
3. Comparacion baseline vs H2O.
4. Seleccion automatica de candidato no-ensemble con restriccion de degradacion RMSE.
5. NSGA-II sobre validacion interna por `unit`.
6. Evaluacion final en test oficial de soluciones no dominadas encontradas en validacion.
7. Comparacion final incluyendo NSGA-II.

### Ubicacion de resultados

- Cada corrida se guarda en `results/runs/{run_id}/`.
- Copia de conveniencia de ultimos resultados en `results/latest/` por tipo de ejecucion.

## Generacion de figuras finales

```powershell
python run_generate_figures.py --results-dir results/final
```
