from src.utils.oi_utils import cargar_dataset, guardar_dataset
from src.utils.data_utils import eliminar_columnas
import pandas as pd

def calcular_columnas(df: pd.DataFrame)->pd.DataFrame:
    
    df = df[
        (df["IMPUESTO_PREDIAL_BRUTO_2021"] != 0) &
        (df["AVALUO_CATASTRAL_2020"] != 0) &
        (df["TARIFA_PREDIAL_2020"] != 0)
    ]
    # Crear la columna 'Descuento' (Predial Bruto - Predial Aplicado)
    df["DESCUENTO"] = (df["IMPUESTO_PREDIAL_BRUTO_2021"] - df["IMPUESTO_PREDIAL_APLICADO_2021"])/df[
        "IMPUESTO_PREDIAL_BRUTO_2021"]

    # Crear la columna 'Variación Avalúo' (Año actual - Año anterior) / Año anterior
    df["VARIACION_AVALUO_2021"] = (df["AVALUO_CATASTRAL_2021"] - df["AVALUO_CATASTRAL_2020"]) / df[
        "AVALUO_CATASTRAL_2020"]

    # Crear la columna 'Variación Tarifa' (Año actual - Año anterior) / Año anterior
    df["VARIACION_TARIFA_2021"] = (df["TARIFA_PREDIAL_2021"] - df["TARIFA_PREDIAL_2020"]) / df[
        "TARIFA_PREDIAL_2020"]

    return df

def run(carpeta: str, zona: str):

    input_path = f"data/processed/{carpeta}/3_Dataset_{zona}.csv"
    output_path = f"data/processed/{carpeta}/4_Dataset_{zona}_col_cal_2.csv"

    df = cargar_dataset(input_path)

    df = calcular_columnas(df)

    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    if not nulos.empty:
        print("Columnas con valores nulos y su cantidad:")
        print(nulos)
    else:
        print("No hay columnas con valores nulos.")

    input("Presiona Enter para continuar...")

    df = eliminar_columnas(df, ['IMPUESTO_PREDIAL_BRUTO_2021','IMPUESTO_PREDIAL_APLICADO_2021',
                                'AVALUO_CATASTRAL_2020', 'AVALUO_CATASTRAL_2021',
                                'TARIFA_PREDIAL_2021', 'TARIFA_PREDIAL_2020',
                                'IMPUESTO_PREDIAL_BRUTO_2019', 'IMPUESTO_PREDIAL_BRUTO_2020',
                                'IMPUESTO_PREDIAL_APLICADO_2019', 'IMPUESTO_PREDIAL_APLICADO_2020', 
                                'TARIFA_PREDIAL_2019'])

    # Obtener el orden de las columnas, colocando "COMPORTAMIENTO_PAGO" al final
    columnas_ordenadas = [col for col in df.columns if col != "COMPORTAMIENTO_PAGO"] + ["COMPORTAMIENTO_PAGO"]

    # Reordenar el DataFrame
    df = df[columnas_ordenadas]

    guardar_dataset(df, output_path)
