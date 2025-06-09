from src.data.io_utils import cargar_dataset, guardar_dataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

def escalar_datos(dataset: str, carpeta: str, metodo='zscore'):
    input_path = f"data/processed/{carpeta}/Dataset_{dataset}.csv"
    output_path = f"data/processed/{carpeta}/Dataset_{dataset}_estandarizado.csv"

    df = cargar_dataset(input_path)

    if metodo == 'zscore':
        scaler = StandardScaler()
    elif metodo == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Método debe ser 'zscore' o 'robust'")

    x = df.drop(['COMPORTAMIENTO_PAGO', 'NUMPRED', 'BARRIO'], axis=1)
    y = df[['COMPORTAMIENTO_PAGO']]  # mantener en formato DataFrame

    columnas_passthrough = ['NUMPRED', 'BARRIO']
    columnas_escaladas = x.columns.tolist()
    columnas_finales = columnas_escaladas + columnas_passthrough + ['COMPORTAMIENTO_PAGO']

    ct = ColumnTransformer(
        transformers=[('scaler', scaler, columnas_escaladas)],
        remainder='passthrough'
    )

    datos_escalados = ct.fit_transform(df.drop(columns=['COMPORTAMIENTO_PAGO']))

    df_escalado = pd.DataFrame(datos_escalados, columns=columnas_escaladas + columnas_passthrough)

    # Añadir nuevamente la variable objetivo
    df_escalado['COMPORTAMIENTO_PAGO'] = y.values

    # Reordenar columnas
    df_escalado = df_escalado[columnas_finales]

    # Guardar
    guardar_dataset(df_escalado, output_path)

    return df_escalado
