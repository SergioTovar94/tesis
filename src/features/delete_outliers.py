import os
import pandas as pd
from src import config
from src.data.io_utils import cargar_dataset, guardar_dataset

def eliminar_outliers(df: pd.DataFrame)-> pd.DataFrame:
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    columnas_numericas = columnas_numericas[
        ~columnas_numericas.str.contains('|'.join(config.COLUMNAS_EXCLUIR_OUTLIERS))]
    print(columnas_numericas)

    filtro_valido = pd.Series(True, index=df.index)

    # Eliminar outliers con el criterio IQR
    for col in columnas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        LIM_INF = Q1 - config.IQR_FACTOR * IQR
        LIM_SUP = Q3 + config.IQR_FACTOR * IQR

        filtro_col = (df[col] >= LIM_INF) & (df[col] <= LIM_SUP)
        # Combinar el filtro con AND para mantener solo filas válidas en todas las columnas
        filtro_valido = filtro_valido & filtro_col

    df_filtrado = df[filtro_valido].copy()
    return df_filtrado

def run(dataset: str):
    input_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, dataset)
    nombre_sin_ext = os.path.splitext(dataset)[0]
    output_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, nombre_sin_ext, '_sin_outliers.csv')

    df = cargar_dataset(input_path)

    df = eliminar_outliers(df)

    guardar_dataset(df, output_path)