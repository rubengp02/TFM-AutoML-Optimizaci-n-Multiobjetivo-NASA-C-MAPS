"""Utilidades para cargar y preparar subconjuntos NASA C-MAPSS."""

# IMPORTS
from __future__ import annotations

from pathlib import Path

import pandas as pd


# LOCALIZACIÓN DEL PROYECTO

def get_project_root(start_path: Path | None = None) -> Path:
    """Devuelve la raíz del proyecto buscando hacia arriba el directorio data."""
    current_path = (start_path or Path.cwd()).resolve() # Escoge el directorio actual o uno dado
    # Recorre la carpeta escogida y todas las superiores buscando una carpeta "data".
    for candidate_path in [current_path, *current_path.parents]:
        if (candidate_path / "data").exists():
            return candidate_path
    raise FileNotFoundError("Project root not found. Expected a 'data' directory.")


# DEFINICIÓN DE COLUMNAS Y ARCHIVOS C-MAPSS

def get_cmapss_column_names() -> list[str]:
    """Construye los nombres de las columnas de C-MAPSS."""
    op_settings = [f"op_setting_{i}" for i in range(1, 4)] # Estas son las columnas de condiciones operativas (op_setting_1, op_setting_2, op_setting_3)
    sensors = [f"sensor_{i}" for i in range(1, 22)] # Estas son las columnas de sensores (sensor_1, sensor_2, ..., sensor_21)
    return ["unit", "cycle", *op_settings, *sensors]


# Para cada subconjunto (FD001, FD002, FD003, FD004) hay tres archivos: train, test y RUL. Esta función construye los nombres de archivo esperados para un subconjunto dado.
def get_subset_filenames(subset: str) -> dict[str, str]:
    """Construye los nombres de archivo esperados para un subconjunto dado (p. ej., FD001, FD004)."""
    subset_norm = subset.upper()
    return {
        "train": f"train_{subset_norm}.txt",
        "test": f"test_{subset_norm}.txt",
        "rul": f"RUL_{subset_norm}.txt",
    }


# BÚSQUEDA Y CARGA DE ARCHIVOS

def find_subset_files(raw_dir: Path, subset: str) -> dict[str, Path]:
    """Busca recursivamente los archivos train/test/RUL del subconjunto solicitado.
    
    Si se busca FD004, se espera que la función devuelva la ruta de los archivos train_FD004.txt, test_FD004.txt y RUL_FD004.txt.
    """
    filenames = get_subset_filenames(subset) # Se aplica la función anterior, que devuelve un diccionario con los nombres de archivos que se buscan.
    subset_file_paths = {} # Aqui se guardarán las rutas encontradas de cada archivo

    # Se recorre el diccionario, buscando en raw_dir el archivo correspondiente. Si no se encuentra, se lanza un error.
    for key, filename in filenames.items():
        matches = list(raw_dir.rglob(filename))
        if not matches:
            raise FileNotFoundError(f"Could not find '{filename}' under: {raw_dir}")
        if len(matches) > 1: # Si se encuentran conincidencias,e lanza un error.
            raise ValueError(f"Multiple matches for '{filename}': {matches}")
        subset_file_paths[key] = matches[0] # Se guarda la ruta encontrada en el diccionario de resultados.

    return subset_file_paths


def load_cmapss_table(file_path: Path, columns: list[str]) -> pd.DataFrame:
    """Carga la tabla train/test de C-MAPSS en Dataframe."""
    return pd.read_csv(
        file_path,
        sep=r"\s+", # Como las columnas de C-MAPSS están separadas por espacios, este es el separador a usar
        header=None,
        names=columns,
        engine="python",
    )


def load_cmapss_rul(file_path: Path) -> pd.Series:
    """Carga el archivo RUL y se queda con la primera columna, la de RUL."""
    rul_series = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python").iloc[:, 0]
    return rul_series.reset_index(drop=True).rename("RUL")


# PREPARACIÓN DE LA VARIABLE OBJETIVO

def add_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el RUL de train como max_cycle_por_unidad - ciclo_actual."""
    train_with_rul_df = train_df.copy()
    train_with_rul_df["RUL"] = train_with_rul_df.groupby("unit")["cycle"].transform("max") - train_with_rul_df["cycle"] # Se agrupa por "unit" y se calcula el ciclo máximo para cada unidad, luego se resta el ciclo actual para obtener el RUL.
    return train_with_rul_df


def get_test_last_observation(test_df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve la fila del último ciclo de cada unidad en test."""
    return (
        test_df.sort_values(["unit", "cycle"]) 
        .groupby("unit", as_index=False)
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )# Ordena por unidad y ciclo.
     # Luego, se queda con la ultima fila de cada motor (ultimo ciclo).
     # Por último, se ordena por unidad y se resetea el índice.


# PIPELINE PRINCIPAL DE PREPARACIÓN DEL DATASET

def prepare_cmapss_data(subset: str = "FD001", raw_dir: Path | None = None) -> dict[str, pd.DataFrame | pd.Series | dict[str, Path] | str]:
    """Carga y prepara un subconjunto C-MAPSS para modelado baseline."""
    subset_norm = subset.upper() # Normaliza el nombre del subset a mayúsculas.
    project_root = get_project_root() # Obtiene la raíz del proyecto.
    base_raw = raw_dir or (project_root / "data" / "raw")

    files = find_subset_files(base_raw, subset=subset_norm) # Busca los archivos correspondientes según el subset solicitado.
    columns = get_cmapss_column_names() # Obtiene los nombres de las columnas para cargar los archivos train/test.

    train_df = load_cmapss_table(files["train"], columns) # Carga el archivo de entrenamiento en un DataFrame.
    test_df = load_cmapss_table(files["test"], columns) # Carga el archivo de test en un DataFrame.
    y_test = load_cmapss_rul(files["rul"]) # Carga el archivo de RUL en una Serie.

    train_df = add_train_rul(train_df) # Añade la columna de RUL al DataFrame de entrenamiento.
    test_last_df = get_test_last_observation(test_df) # Obtiene la última observación de cada unidad en el DataFrame de test.

    if len(test_last_df) != len(y_test):
        raise ValueError(
            f"Number of test engines ({len(test_last_df)}) does not match RUL labels ({len(y_test)})."
        )

    return {
        "subset": subset_norm,
        "files": files,
        "train_df": train_df,
        "test_df": test_df,
        "test_last_df": test_last_df,
        "y_test": y_test,
    } # Devuelve un diccionario con toda la información cargada y preparada para el subset solicitado.
