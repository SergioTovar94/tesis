from src.data.io_utils import cargar_dataset, guardar_dataset
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
import os

def escalar_datos(carpeta: str, dataset: str, metodo='zscore'):
    input_path = f"data/processed/{carpeta}/{dataset}"
    nombre_sin_ext = os.path.splitext(dataset)[0]
    output_path = f"data/processed/{carpeta}/{nombre_sin_ext}_estandarizado.csv"

    df = cargar_dataset(input_path)

    # Seleccionar método de escalamiento
    if metodo == 'zscore':
        scaler = StandardScaler()
    elif metodo == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Método debe ser 'zscore' o 'robust'")

    # Seleccionar columnas numéricas a escalar
    columnas_numericas = df.select_dtypes(include=['number']).columns.difference(['NUMPRED', 'COMPORTAMIENTO_PAGO'])
    X = df[columnas_numericas]
    y = df['COMPORTAMIENTO_PAGO']
    numpred = df['NUMPRED'] if 'NUMPRED' in df.columns else None

    # Escalar
    X_scaled = scaler.fit_transform(X)
    df_escalado = pd.DataFrame(X_scaled, columns=columnas_numericas)

    # Agregar columnas no escaladas
    if numpred is not None:
        df_escalado['NUMPRED'] = numpred.values
    df_escalado['COMPORTAMIENTO_PAGO'] = y.values

    # Reordenar columnas
    columnas_finales = (['NUMPRED'] if numpred is not None else []) + columnas_numericas.tolist() + ['COMPORTAMIENTO_PAGO']
    df_escalado = df_escalado[columnas_finales]

    # Guardar resultado
    guardar_dataset(df_escalado, output_path)

    return df_escalado
