import pandas as pd
import logging
from colorama import Fore, Style

def eliminar_columnas(df: pd.DataFrame, columnas_a_eliminar: list) -> pd.DataFrame:
    """Elimina columnas innecesarias."""
    df = df.drop(columns=columnas_a_eliminar)
    logging.info(f"✅ Se eliminaron las columnas: {columnas_a_eliminar}")
    return df

def eliminar_nulos(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Elimina registros con valores nulos en la columna indicada."""
    nulos = df[columna].isna().sum()
    df = df.dropna(subset=[columna])
    logging.info(f"✅ Se eliminaron {Fore.RED}{nulos}{Style.RESET_ALL} registros con {columna} nulo")
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