from src.data.io_utils import cargar_dataset, guardar_dataset
from src.utils.print_utils import print_message

import pandas as pd
import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

def pivotear(df: pd.DataFrame, columnas_fijas: list, pivote: list, columnas_a_pivotear: list) -> pd.DataFrame:
    df_filtrado = df[columnas_fijas + pivote + columnas_a_pivotear]
    df_transversal = df_filtrado.pivot(index=columnas_fijas, columns=pivote[0], values=columnas_a_pivotear)
    df_transversal.columns = [f"{col[0]}_{col[1]}" for col in df_transversal.columns]
    df_transversal = df_transversal.reset_index()
    return df_transversal

def run(carpeta: str):
    print_message("Pasando de panel a transversal")
    input_path = f"data/processed/{carpeta}/Dataset_panel_depurada.csv"
    output_path = f"data/processed/{carpeta}/Dataset_transversal.csv"

    # 1. Cargar datos desde data/raw/
    df = cargar_dataset(input_path)

    # 2. Transformar de panel a transversal
    columnas_fijas = ['NUMPRED', 'TIPO_PREDIO', 'ESTRATO', 'AREA_CONSTRUIDA']
    columnas_a_pivotear = ['AVALUO_CATASTRAL','IMPUESTO_PREDIAL_BRUTO', 'IMPUESTO_PREDIAL_APLICADO', 'RECIPU']
    pivote = ['VIGENCIA']
    df_transversal = pivotear(df, columnas_fijas, pivote, columnas_a_pivotear)
    
    # 3. Guardar el resultado en data/processed/
    guardar_dataset(df_transversal, output_path)
    print(f"✅ Base panel convertida a transversal")
