import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from src.data.io_utils import cargar_dataset, guardar_dataset
import os

def aplicar_smote(carpeta: str, dataset: str):
    input_path = f"data/processed/{carpeta}/{dataset}"

    nombre_sin_ext = os.path.splitext(dataset)[0]

    output_path = f"data/processed/{carpeta}/{nombre_sin_ext}_smote.csv"

    df = cargar_dataset(input_path)

    columnas_numericas = df.select_dtypes(include=['number']).columns.difference(['NUMPRED', 'COMPORTAMIENTO_PAGO'])
    X = df[columnas_numericas]
    y = df['COMPORTAMIENTO_PAGO']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name='COMPORTAMIENTO_PAGO')], axis=1)

    guardar_dataset(df, output_path)
