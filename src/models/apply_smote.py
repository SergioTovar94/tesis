from src.data.io_utils import cargar_dataset, guardar_dataset
from src.utils.data_utils import imprimir_nulos
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

def sobremuestreo(x: pd.DataFrame, y: list)->pd.DataFrame:
    smote = SMOTE(random_state=42)
    x_smote, y_smote = smote.fit_resample(x, y)
    return pd.concat([pd.DataFrame(x_smote), pd.Series(y_smote, name='COMPORTAMIENTO_PAGO')], axis=1)

def sumbuestreo(x: pd.DataFrame, y: list) -> pd.DataFrame:
    undersampler = RandomUnderSampler(random_state=42)
    x_under, y_under = undersampler.fit_resample(x, y)
    return pd.concat([pd.DataFrame(x_under), pd.Series(y_under, name='COMPORTAMIENTO_PAGO')], axis=1)

def hibrido(x: pd.DataFrame, y: list) -> pd.DataFrame:
    smote_tomek = SMOTETomek(random_state=42)
    x_hybrid, y_hybrid = smote_tomek.fit_resample(x, y)
    return pd.concat([pd.DataFrame(x_hybrid), pd.Series(y_hybrid, name='COMPORTAMIENTO_PAGO')], axis=1)

def run(dataset: str, carpeta: str):
    print(f"Aplicando SMOTE a dataset {dataset}")
    input_path = f"data/processed/{carpeta}/Dataset_{dataset}.csv"

    df = cargar_dataset(input_path)

    # Separar características (X) y etiquetas (y)
    x = df.drop(['COMPORTAMIENTO_PAGO'], axis=1)
    imprimir_nulos(x)
    y = df['COMPORTAMIENTO_PAGO']
    print(x.dtypes)
    # 1. Sobremuestreo (SMOTE)
    data_sobremuestreo = sobremuestreo(x, y)
    data_submuestreo = sumbuestreo(x, y)
    data_hibrido = hibrido(x, y)

    output_path = f"data/processed/{carpeta}/Dataset_{dataset}_smote.csv"
    guardar_dataset(data_sobremuestreo, output_path)

    output_path = f"data/processed/{carpeta}/Dataset_{dataset}_undersample.csv"
    guardar_dataset(data_submuestreo, output_path)

    output_path = f"data/processed/{carpeta}/Dataset_{dataset}_hibrido.csv"
    guardar_dataset(data_hibrido, output_path)

    print("Equilibrado realizado exitosamente:")

