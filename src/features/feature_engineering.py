from src.preprocessing.clean_data import eliminar_ceros_df
from src.utils.data_utils import reposiconar_comportamiento_pago
import pandas as pd

def calcular_columnas(df: pd.DataFrame)->pd.DataFrame:
    """
    Calcula y agrega columnas derivadas al DataFrame:
    - DESCUENTO
    - VARIACION_AVALUO
    - VARIACION_TARIFA

    Retorna el DataFrame actualizado.
    """
    df = calcular_variacion(df, "DESCUENTO", "IMPUESTO_PREDIAL_BRUTO_2021", "IMPUESTO_PREDIAL_APLICADO_2021")
    df = calcular_variacion(df, "VARIACION_AVALUO", "AVALUO_CATASTRAL_2021", "AVALUO_CATASTRAL_2020")
    df = calcular_variacion(df, "VARIACION_TARIFA", "TARIFA_PREDIAL_2021", "TARIFA_PREDIAL_2020")

    return df

def calcular_variacion(df: pd.DataFrame, nueva_col: str, col1: str, col2: str) -> pd.DataFrame:
    """
    Crea una nueva columna como la diferencia relativa entre dos columnas existentes.
    nueva_col = (col1 - col2) / col2
    """
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columnas {col1} o {col2} no existen en el DataFrame.")
    df[nueva_col] = (df[col1] - df[col2]) / df[col2]
    return df

def run(tipo: str, carpeta: str):
    """
    Ejecuta la función de ingeniería de características.
    """
    print("Ejecutando feature engineering")
    input_path = f"data/processed/{carpeta}/Dataset_{tipo}.csv"
    output_path = f"data/processed/{carpeta}/Dataset_{tipo}_col_cal.csv"

    df = pd.read_csv(input_path)
    df = eliminar_ceros_df(df)
    df = calcular_columnas(df)
    df = reposiconar_comportamiento_pago(df)
    df.to_csv(output_path, index=False)

