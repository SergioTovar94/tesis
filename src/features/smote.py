import os
from src import config
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from src.utils.io_utils import cargar_dataset, guardar_dataset

def run(dataset: str):
    input_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, dataset)

    nombre_sin_ext = os.path.splitext(dataset)[0]
    output_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, f'{nombre_sin_ext}_smote.csv')

    df = cargar_dataset(input_path)

    columnas_numericas = df.select_dtypes(include=['number']).columns.difference([config.LLAVE, config.VARIABLE_OBJETIVO])
    X = df[columnas_numericas]
    y = df[config.VARIABLE_OBJETIVO]

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=config.VARIABLE_OBJETIVO)], axis=1)

    guardar_dataset(df, output_path)
