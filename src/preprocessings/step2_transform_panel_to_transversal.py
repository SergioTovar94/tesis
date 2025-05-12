from src.utils.oi_utils import cargar_dataset, guardar_dataset
from src.utils.data_utils import eliminar_columnas
import pandas as pd
import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

def pivotear(df: pd.DataFrame, columnas_fijas: list, pivote: list, columnas_a_pivotear: list) -> pd.DataFrame:
    df_filtrado = df[columnas_fijas + pivote + columnas_a_pivotear]
    df_transversal = df_filtrado.pivot(index=columnas_fijas, columns=pivote[0], values=columnas_a_pivotear)
    df_transversal.columns = [f"{col[0]}_{col[1]}" for col in df_transversal.columns]
    df_transversal = df_transversal.reset_index()
    return df_transversal

def calcular_tarifa(avaluo):
    salario_minimo = 908526
    if avaluo < 27 * salario_minimo:
        return 0.005
    elif avaluo < 62 * salario_minimo:
        return 0.006
    elif avaluo < 135 * salario_minimo:
        return 0.007
    elif avaluo < 180 * salario_minimo:
        return 0.008
    elif avaluo < 269 * salario_minimo:
        return 0.009
    elif avaluo < 414 * salario_minimo:
        return 0.010
    else:
        return 0.016

def run(carpeta: str):

    input_path = f"data/processed/{carpeta}/1_Dataset_panel_depurada.csv"
    output_path = f"data/processed/{carpeta}/2_Dataset_transversal.csv"

    # 1. Cargar datos desde data/raw/
    df = cargar_dataset(input_path)

    # 2. Transformar de panel a transversal
    columnas_fijas = ['NUMPRED', 'TIPO_PREDIO', 'ESTRATO', 'AREA_CONSTRUIDA']
    columnas_a_pivotear = ['AVALUO_CATASTRAL', "TARIFA_PREDIAL",'IMPUESTO_PREDIAL_BRUTO', 'IMPUESTO_PREDIAL_APLICADO',
                           'RECIPU']
    pivote = ['VIGENCIA']
    df_transversal = pivotear(df, columnas_fijas, pivote, columnas_a_pivotear)

    df = eliminar_columnas(df_transversal, ['TARIFA_PREDIAL_2021'])
    df['TARIFA_PREDIAL_2021'] = df['AVALUO_CATASTRAL_2021'].apply(calcular_tarifa)

    # 3. Guardar el resultado en data/processed/
    guardar_dataset(df, output_path)
    print(f"✅ Base panel convertida a transversal")
