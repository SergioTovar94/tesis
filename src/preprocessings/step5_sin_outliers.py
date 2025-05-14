from src.utils.oi_utils import cargar_dataset, guardar_dataset
import pandas as pd
from colorama import Fore, Style

def eliminar_outliers(df: pd.DataFrame)-> pd.DataFrame:
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    columnas_numericas = columnas_numericas[~columnas_numericas.str.contains('RECIPU')]
    columnas_numericas = columnas_numericas[~columnas_numericas.str.contains('ANIOS')]

    columnas_numericas = columnas_numericas[
        (columnas_numericas != 'ESTRATO') & 
        (~columnas_numericas.str.contains('TARIFA', case=False))
    ]

    print(columnas_numericas)
    # Eliminar outliers con el criterio IQR
    for col in columnas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        LIM_INF = Q1 - 1.5 * IQR
        LIM_SUP = Q3 + 1.5 * IQR
        df = df[(df[col] >= LIM_INF) & (df[col] <= LIM_SUP)]
    return df

def run(dataset: str, carpeta: str):

    input_path = f"data/processed/{carpeta}/{dataset}.csv"
    output_path = f"data/processed/{carpeta}/{dataset}_sin_outliers.csv"

    df = cargar_dataset(input_path)

    df = eliminar_outliers(df)

    guardar_dataset(df, output_path)
