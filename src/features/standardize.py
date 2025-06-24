import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from src import config
from src.data.io_utils import cargar_dataset, guardar_dataset

def escalar_datos(dataset: str, metodo: str):
    input_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, dataset)
    nombre_sin_ext = os.path.splitext(dataset)[0]
    output_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, nombre_sin_ext, '_estandarizado.csv')
    df = cargar_dataset(input_path)

    # Seleccionar método de escalamiento
    if metodo == 'zscore':
        scaler = StandardScaler()
    elif metodo == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Método debe ser 'zscore' o 'robust'")

    # Seleccionar columnas numéricas a escalar
    columnas_numericas = df.select_dtypes(include=['number']).columns.difference([config.LLAVE, config.VARIABLE_OBJETIVO])
    X = df[columnas_numericas]
    y = df[config.VARIABLE_OBJETIVO]
    numpred = df[config.LLAVE] if config.LLAVE in df.columns else None

    # Escalar
    X_scaled = scaler.fit_transform(X)
    df_escalado = pd.DataFrame(X_scaled, columns=columnas_numericas)

    # Agregar columnas no escaladas
    if numpred is not None:
        df_escalado[config.LLAVE] = numpred.values
    df_escalado[config.VARIABLE_OBJETIVO] = y.values

    # Reordenar columnas
    columnas_finales = ([config.LLAVE] if numpred is not None else []) + columnas_numericas.tolist() + [config.VARIABLE_OBJETIVO]
    df_escalado = df_escalado[columnas_finales]

    # Guardar resultado
    guardar_dataset(df_escalado, output_path)

    return df_escalado
