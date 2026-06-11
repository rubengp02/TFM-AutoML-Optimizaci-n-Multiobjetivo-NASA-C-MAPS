# TFM - AutoML y optimización multiobjetivo para predicción de RUL

Repositorio asociado al Trabajo Fin de Máster:

**Optimización multiobjetivo de modelos de aprendizaje automático mediante técnicas de Automated Machine Learning (AutoML) para la predicción de vida útil restante en mantenimiento predictivo**.

El proyecto implementa un flujo experimental completo para la predicción de la vida útil restante, o **Remaining Useful Life (RUL)**, sobre el conjunto de datos **NASA C-MAPSS**, utilizando modelos tradicionales de aprendizaje automático, H2O AutoML y optimización multiobjetivo mediante NSGA-II.

## Objetivo del proyecto

El objetivo principal es comparar distintos enfoques de modelado no solo en términos de precisión predictiva, sino también considerando métricas de eficiencia computacional.

Los criterios utilizados en la comparación son:

- RMSE en el conjunto de test oficial.
- Tiempo de entrenamiento del modelo.
- Tiempo de inferencia.
- Tiempo total asociado al proceso de búsqueda o selección del modelo.

El experimento final se centra en el subconjunto **FD004** de NASA C-MAPSS.

## Estructura del repositorio

```text
.
├─ README.md
├─ requirements.txt
├─ environment_versions.txt
├─ paso_01_entrenar_modelos_tradicionales.py
├─ paso_02_entrenar_h2o_automl.py
├─ paso_03_comparacion_inicial.py
├─ paso_04_seleccionar_compromiso_pareto.py
├─ paso_05_ejecutar_nsga2.py
├─ paso_06_comparacion_final.py
├─ paso_07_generar_figuras_finales.py
├─ src/
└─ results/
   └─ runs/
      ├─ FD004_baseline/
      ├─ FD004_h2o_30models/
      ├─ FD004_comparison_initial/
      ├─ FD004_compromise_selection/
      ├─ FD004_nsga2_60trials/
      └─ FD004_comparison_final/