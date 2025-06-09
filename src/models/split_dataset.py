from src.data.io_utils import cargar_dataset, guardar_dataset
from utils.data_utils import eliminar_columnas
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd

def seleccionar_columnas_calculadas(df):
    columnas_deseadas = [
        'ESTRATO',
        'AREA_CONSTRUIDA',
        'ANIOS_PAGADOS',
        'DESCUENTO',
        'VARIACION_AVALUO',
        'VARIACION_TARIFA',
        'COMPORTAMIENTO_PAGO'
    ]
    
    # Filtra el DataFrame y devuelve solo esas columnas
    return df[columnas_deseadas].copy()

def run(dataset: str, carpeta: str):

    input_path = f"data/processed/{carpeta}/Dataset_{dataset}.csv"    

    df = cargar_dataset(input_path)
    df_2 = seleccionar_columnas_calculadas(df)
    train_df, test_df = train_test_split(df_2, test_size=0.3, random_state=42, shuffle=True)
    train_df = eliminar_columnas(train_df, ['ESTRATO'])
    test_df = eliminar_columnas(test_df, ['ESTRATO'])
    output_path = f"data/processed/{carpeta}/Entrenamiento_dataset_{dataset}.csv"
    guardar_dataset(train_df, output_path)
    output_path = f"data/processed/{carpeta}/Evaluacion_dataset_{dataset}.csv"
    guardar_dataset(test_df, output_path)

    print("¡Listo! Archivos guardados como 'entrenamiento.csv' y 'evaluacion.csv'")
