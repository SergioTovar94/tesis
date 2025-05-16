from src.data.oi_utils import cargar_dataset, guardar_dataset
from src.utils.data_utils import eliminar_columnas
from src.features.feature_engineering import calcular_columnas

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