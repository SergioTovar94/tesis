import os
import pandas as pd
import logging
from src import config
from src.data.io_utils import cargar_dataset, guardar_dataset
from src.utils.print_utils import print_message
from src.utils.data_utils import eliminar_nan_df, eliminar_ceros_df

def calcular_tarifa(avaluo: float, año: int) -> float:
    """
    Calcula la tarifa predial según el avalúo y el año (estatuto vigente).
    Puedes modificar la lógica según los rangos del estatuto real.

    Args:
        avaluo (float): Avalúo catastral del predio.
        año (int): Año a considerar para aplicar el estatuto correspondiente.

    Returns:
        float: Tarifa predial aplicada.
    """
    if año == 2019:
        salario_minimo = config.SALARIO_MINIMO_2019
    elif año == 2020:
        salario_minimo = config.SALARIO_MINIMO_2020
    else:
        salario_minimo = config.SALARIO_MINIMO_2021
        
    if año <= 2020:
        tarifas = config.TARIFAS_ANTIGUAS
    else:
        tarifas = config.TARIFAS_NUEVAS
    
    for limite, tarifa in tarifas:
        if avaluo < limite * salario_minimo:
            return tarifa

def corregir_estratos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige valores atípicos de estrato (0, 8, 9) reemplazándolos por la moda del barrio.
    Si el barrio no tiene datos válidos, usa la moda global.
    
    Args:
        df: DataFrame con columnas 'BARRIO' y 'ESTRATO'
        
    Returns:
        DataFrame con la columna 'ESTRATO' corregida
    """
    print_message("Corrigiendo estratos atípicos...")
    
    # 1. Calcular la moda por barrio
    moda_por_barrio = (
        df[~df['ESTRATO'].isin(config.ESTRATOS_ATIPICOS)]
        .groupby('BARRIO')['ESTRATO']
        .agg(lambda x: x.mode()[0])
    )
    
    # 2. Calcular la moda global para barrios no encontrados
    moda_global = df[~df['ESTRATO'].isin(config.ESTRATOS_ATIPICOS)]['ESTRATO'].mode()[0]
    
    # 3. Convertir a diccionario para mejor performance
    moda_dict = moda_por_barrio.to_dict()
    
    # 4. Aplicar la corrección
    df['ESTRATO'] = df.apply(
        lambda row: moda_dict.get(row['BARRIO'], moda_global) 
                   if row['ESTRATO'] in config.ESTRATOS_ATIPICOS 
                   else row['ESTRATO'],
        axis=1
    )
    
    logging.info(f"Estratos corregidos. Moda global: {moda_global}")
    return df

def run():
    """
    Ejecuta la limpieza de datos en el DataFrame.

    Args:
        carpeta (str): Ruta de la carpeta donde se encuentran los archivos CSV.
    """
    print_message("Limpiando datos")
    # Cargar el DataFrame desde un archivo CSV
    input_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, config.TRANSVERSAL)
    output_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, config.TRANSVERSAL_DEPURADA)

    df = cargar_dataset(input_path)

    # Eliminar registros con NaN
    df = eliminar_nan_df(df)

    # Eliminar registros con ceros en columnas específicas
    df = eliminar_ceros_df(df)
    df["TARIFA_PREDIAL_2019"] = df.apply(lambda row: calcular_tarifa(row["AVALUO_CATASTRAL_2019"], 2019), axis=1)
    df["TARIFA_PREDIAL_2020"] = df.apply(lambda row: calcular_tarifa(row["AVALUO_CATASTRAL_2020"], 2020), axis=1)
    df["TARIFA_PREDIAL_2021"] = df.apply(lambda row: calcular_tarifa(row["AVALUO_CATASTRAL_2021"], 2021), axis=1)
    
    
    # Guardar el DataFrame limpio en un nuevo archivo CSV
    guardar_dataset(df, output_path)
    print(f"✅ Base panel convertida a transversal")