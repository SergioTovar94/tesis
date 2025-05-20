import pandas as pd
import logging
from colorama import Fore, Style
from src.data.io_utils import cargar_dataset, guardar_dataset

def eliminar_columnas(df: pd.DataFrame, columnas_a_eliminar: list) -> pd.DataFrame:
    """Elimina columnas innecesarias."""
    df = df.drop(columns=columnas_a_eliminar)
    logging.info(f"✅ Se eliminaron las columnas: {columnas_a_eliminar}")
    return df

def imprimir_nulos(df: pd.DataFrame) -> pd.DataFrame:
    nulos = df.isnull().sum()
    columnas_con_nulos = nulos[nulos > 0]
    print("Columnas con valores nulos:")
    print(columnas_con_nulos)

def filtrar(df: pd.DataFrame, col: str, valores_a_eliminar: list) -> pd.DataFrame:
    """Elimina predios con destinos económicos no deseados."""
    num_registros_antes = len(df)
    df = df[~df[col].isin(valores_a_eliminar)]
    registros_eliminados = num_registros_antes - len(df)
    logging.info(f"✅ Se eliminaron {Fore.RED}{registros_eliminados}{Style.RESET_ALL} predios con destino: {valores_a_eliminar}")
    return df

def filtrar_por_anio(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    columnas_a_eliminar = [col for col in df.columns if f"_{anio}]" in col]
    df = df.drop(columns=columnas_a_eliminar, errors='ignore')
    print(f"✅ Registros anio {anio} eliminados")
    return df

def reposiconar_comportamiento_pago(df: pd.DataFrame) -> pd.DataFrame: 
    """Reubica el comportamiento de pago en la columna 'comportamiento_pago'."""
    col = 'COMPORTAMIENTO_PAGO'
    comportamiento = df.pop(col)
    df[col] = comportamiento
    return df