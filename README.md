# TFM - AutoML y optimización multiobjetivo para predicción de RUL

Repositorio asociado al Trabajo Fin de Máster:

**Optimización multiobjetivo de modelos de aprendizaje automático mediante técnicas de Automated Machine Learning (AutoML) para la predicción de vida útil restante en mantenimiento predictivo**.

El proyecto desarrolla un flujo experimental completo para la predicción de la vida útil restante, o **Remaining Useful Life (RUL)**, sobre el conjunto de datos **NASA C-MAPSS**, utilizando modelos tradicionales de aprendizaje automático, H2O AutoML y optimización multiobjetivo mediante NSGA-II.

El repositorio recoge la versión final del código utilizado en la memoria, junto con los resultados consolidados de la ejecución experimental definitiva.

## Objetivo del proyecto

El objetivo del trabajo es comparar distintos enfoques de modelado para la predicción de RUL, evaluando no solo la precisión predictiva, sino también el coste computacional asociado a cada alternativa.

Los criterios considerados en la comparación final son:

* RMSE sobre el conjunto de test oficial.
* Tiempo de entrenamiento del modelo.
* Tiempo de inferencia.
* Tiempo total asociado al proceso de búsqueda o selección del modelo.

La experimentación final se centra en el subconjunto **FD004** de NASA C-MAPSS.

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
```

## Datos utilizados

El conjunto de datos original **NASA C-MAPSS** no se incluye directamente en el repositorio.

Para reproducir la ejecución, los archivos del dataset deben descargarse manualmente y colocarse en la siguiente ruta local:

```text
data/raw/CMAPSSData_extracted/
```

Para la ejecución final del TFM se han utilizado los archivos correspondientes al subconjunto FD004:

```text
train_FD004.txt
test_FD004.txt
RUL_FD004.txt
```

La carpeta `data/raw/` está excluida del repositorio mediante `.gitignore`.

## Dependencias y entorno

Las dependencias principales del proyecto se recogen en:

```text
requirements.txt
```

Además, el archivo:

```text
environment_versions.txt
```

documenta las versiones de Python y Java utilizadas en la ejecución definitiva. Java es necesario para la ejecución de H2O AutoML.

## Flujo experimental implementado

El proyecto se organiza en siete pasos principales. Cada paso corresponde a una fase concreta de la experimentación descrita en la memoria.

### Paso 1: entrenamiento de modelos tradicionales

Archivo:

```text
paso_01_entrenar_modelos_tradicionales.py
```

Este paso carga el subconjunto FD004, prepara los datos, realiza la división interna entre entrenamiento y validación por motores completos, entrena los modelos tradicionales y evalúa el resultado final sobre el test oficial.

Modelos considerados:

* Ridge
* Random Forest
* Extra Trees
* HistGradientBoosting
* XGBoost
* LightGBM

Resultados generados:

```text
results/runs/FD004_baseline/
```

### Paso 2: entrenamiento mediante H2O AutoML

Archivo:

```text
paso_02_entrenar_h2o_automl.py
```

Este paso ejecuta H2O AutoML con un máximo de 30 modelos, selecciona el modelo líder del proceso AutoML y lo evalúa sobre el test oficial.

Resultados generados:

```text
results/runs/FD004_h2o_30models/
```

### Paso 3: comparación inicial

Archivo:

```text
paso_03_comparacion_inicial.py
```

Este paso combina los resultados de los modelos tradicionales y de H2O AutoML, genera una comparación inicial y calcula el frente de Pareto inicial.

Resultados generados:

```text
results/runs/FD004_comparison_initial/
```

### Paso 4: selección de solución de compromiso

Archivo:

```text
paso_04_seleccionar_compromiso_pareto.py
```

Este paso selecciona automáticamente una solución de compromiso no ensamblada a partir de la comparación inicial. Esta selección permite decidir la familia de modelos utilizada posteriormente en la optimización mediante NSGA-II.

Resultados generados:

```text
results/runs/FD004_compromise_selection/
```

### Paso 5: optimización mediante NSGA-II

Archivo:

```text
paso_05_ejecutar_nsga2.py
```

Este paso aplica optimización multiobjetivo mediante NSGA-II sobre LightGBM. La ejecución final utiliza 60 ensayos y una población de 12 individuos.

Los objetivos de minimización son:

* RMSE.
* Tiempo de entrenamiento.
* Tiempo de inferencia.

Resultados generados:

```text
results/runs/FD004_nsga2_60trials/
```

### Paso 6: comparación final

Archivo:

```text
paso_06_comparacion_final.py
```

Este paso integra los resultados de los modelos tradicionales, H2O AutoML y NSGA-II. A partir de ellos genera la comparación final y el frente de Pareto definitivo.

Resultados generados:

```text
results/runs/FD004_comparison_final/
```

### Paso 7: generación de figuras finales

Archivo:

```text
paso_07_generar_figuras_finales.py
```

Este paso genera las figuras finales utilizadas en la memoria a partir de los CSV ya calculados. No vuelve a entrenar modelos.

## Módulos auxiliares

La carpeta `src/` contiene los módulos auxiliares utilizados por los siete pasos principales:

* `cmapss_data_preparation.py`: carga del dataset C-MAPSS y construcción de la variable RUL.
* `exploration.py`: exploración básica y diagnóstico de los datos.
* `traditional_models.py`: entrenamiento, evaluación y exportación de modelos tradicionales.
* `h2o_automl_pipeline.py`: entrenamiento y evaluación de H2O AutoML.
* `model_comparison.py`: unión de resultados, rankings, Pareto y generación de figuras.
* `pareto_compromise_selection.py`: selección de soluciones de compromiso.
* `nsga2_multiobjective_optimization.py`: optimización multiobjetivo mediante Optuna NSGA-II.
* `experiment_tracking.py`: gestión de carpetas de ejecución, identificadores de ejecución y archivos `run_config.json`.
* `config.py`: constantes y parámetros compartidos del proyecto.
* `final_figures.py`: generación de figuras finales.

## Resultados incluidos

El repositorio incluye los resultados consolidados de la ejecución final:

```text
results/runs/FD004_baseline/
results/runs/FD004_h2o_30models/
results/runs/FD004_comparison_initial/
results/runs/FD004_compromise_selection/
results/runs/FD004_nsga2_60trials/
results/runs/FD004_comparison_final/
```

Cada carpeta contiene un archivo:

```text
run_config.json
```

Este archivo recoge la configuración utilizada en cada fase de la experimentación. También se incluyen los CSV principales de resultados y las figuras generadas para el análisis final.

## Ejecución final reproducible

Para reproducir la ejecución final, deben instalarse las dependencias, colocar los archivos del subconjunto FD004 en `data/raw/CMAPSSData_extracted/` y ejecutar los pasos en el mismo orden seguido en la memoria:

```powershell
pip install -r requirements.txt

python paso_01_entrenar_modelos_tradicionales.py
python paso_02_entrenar_h2o_automl.py
python paso_03_comparacion_inicial.py
python paso_04_seleccionar_compromiso_pareto.py
python paso_05_ejecutar_nsga2.py
python paso_06_comparacion_final.py
python paso_07_generar_figuras_finales.py
```

Estos comandos reproducen la ejecución final definida en el código, centrada en FD004 y en las configuraciones empleadas en el TFM.

## Nota sobre reproducibilidad

Los resultados pueden variar ligeramente en función del hardware, la carga del sistema y las versiones exactas de las librerías. Por ello, se incluyen:

* `requirements.txt`, con las dependencias principales.
* `environment_versions.txt`, con las versiones de Python y Java.
* `run_config.json`, dentro de cada carpeta de ejecución.
* CSV finales de resultados.
* Figuras finales generadas a partir de los resultados consolidados.
