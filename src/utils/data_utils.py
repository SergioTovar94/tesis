from src import config
import pandas as pd
import logging
from colorama import Fore, Style

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
    col = config.VARIABLE_OBJETIVO
    comportamiento = df.pop(col)
    df[col] = comportamiento
    return df

def eliminar_nan_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permite ver las columnas con valores NaN y eliminar los NaN de la columna seleccionada sobre un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a analizar y limpiar.

    Returns:
        pd.DataFrame: DataFrame limpio según la columna seleccionada.
    """
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if nan_cols.empty:
        print("✅ No hay columnas con valores NaN en el DataFrame.")
        return df

    print("\n🔍 Columnas con valores NaN:")
    for i, (col, count) in enumerate(nan_cols.items(), 1):
        print(f"{i}. {col}: {count} NaN")

    try:
        entrada = input("\n👉 Ingresa los números de las columnas (separados por comas) de las que deseas eliminar los NaN (0 para cancelar): ").strip()
        if entrada == "0":
            print("🔙 Operación cancelada.")
            return df

        indices = [int(i.strip()) for i in entrada.split(",") if i.strip().isdigit()]
        if any(i < 1 or i > len(nan_cols) for i in indices):
            print("❌ Uno o más números están fuera del rango válido.")
            return df

        columnas_a_limpiar = [nan_cols.index[i - 1] for i in indices]
        df = df.dropna(subset=columnas_a_limpiar)
        print(f"✅ Se eliminaron los registros con NaN en las columnas: {', '.join(columnas_a_limpiar)}.")
        return df
    except Exception as e:
        print(f"❌ Error al procesar la entrada: {e}")
        return df

    
def eliminar_ceros_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permite eliminar filas con ceros en las columnas seleccionadas, mostrando las columnas con ceros
    y actualizando después de cada eliminación.

    Args:
        df (pd.DataFrame): DataFrame a analizar y limpiar.

    Returns:
        pd.DataFrame: DataFrame limpio según las columnas seleccionadas.
    """

    while True:
        zero_counts = (df == 0).sum()
        zero_cols = zero_counts[zero_counts > 0]

        if zero_cols.empty:
            print("✅ No quedan columnas con valores 0 en el DataFrame.")
            break

        print("\n🔍 Columnas con valores 0:")
        for i, (col, count) in enumerate(zero_cols.items(), 1):
            print(f"{i}. {col}: {count} ceros")

        try:
            entrada = input("\n👉 Ingresa el número de la columna de la que deseas eliminar las filas con ceros (0 para salir): ").strip()
            if entrada == "0":
                print("🔚 Operación finalizada por el usuario.")
                break

            opcion = int(entrada)
            if 1 <= opcion <= len(zero_cols):
                col_a_limpiar = zero_cols.index[opcion - 1]
                df = df[df[col_a_limpiar] != 0]
                print(f"✅ Se eliminaron las filas con ceros en la columna '{col_a_limpiar}'.")
            else:
                print("❌ Opción fuera de rango.")
        except ValueError:
            print("❌ Entrada no válida. Ingresa un número válido.")

    return df