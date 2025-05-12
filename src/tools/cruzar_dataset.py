import pandas as pd
from src.utils.oi_utils import cargar_dataset, guardar_dataset
import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

def limpiar_caracteres_especiales(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """Elimina comas y caracteres especiales en las columnas de texto."""
    for col in columnas:
        df[col] = df[col].str.replace(',', '', regex=False)
        df[col] = df[col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    logging.info("✅ Se ajustaron registros con caracteres especiales")
    return df

def run(carpeta: str):
    input_path = f"data/processed/{carpeta}/3_Dataset_urbano.csv"
    input_path_2 = f"data/raw/Disuelto_espacial.csv"
    output_path = f"data/processed/{carpeta}/3_Dataset_urbano.csv"
    # Cargar los datasets
    df1 = cargar_dataset(input_path)
    df2 = cargar_dataset(input_path_2)

    df1['NUMPRED'] = df1['NUMPRED'].astype(str)
    df2['COD_NUM'] = df2['COD_NUM'].astype(str)

    # Hacer un LEFT JOIN para ver cuáles de df1 NO están en df2
    df1_join = df1.merge(df2[['COD_NUM']], how='left', left_on='NUMPRED', right_on='COD_NUM')
    no_en_df2 = df1_join[df1_join['COD_NUM'].isna()]

    # Hacer un RIGHT JOIN para ver cuáles de df2 NO están en df1
    df2_join = df2.merge(df1[['NUMPRED']], how='left', left_on='COD_NUM', right_on='NUMPRED')
    no_en_df1 = df2_join[df2_join['NUMPRED'].isna()]

    # Mostrar resultados
    print(f"🔍 Registros en dataset1 que NO están en dataset2: {len(no_en_df2)}")
    print("Ejemplo:")
    print(no_en_df2.head(1)[['NUMPRED']])

    print(f"\n🔍 Registros en dataset2 que NO están en dataset1: {len(no_en_df1)}")
    print("Ejemplo:")
    print(no_en_df1.head(1)[['COD_NUM']])

    # Hacer el left join usando NUMPRED y COD_NUM como clave
    df_resultado = df1.merge(df2, how='left', left_on='NUMPRED', right_on='COD_NUM')

    df = limpiar_caracteres_especiales(df_resultado, ['BARRIO_2'])

    # Mostrar cuántos registros coinciden
    coincidencias = df_resultado['COD_NUM'].notna().sum()
    print(f"Coincidencias encontradas: {coincidencias}")

    # Pausar para revisar si es necesario
    input("Presiona Enter para continuar...")

    guardar_dataset(df_resultado, output_path)
