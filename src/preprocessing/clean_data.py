import pandas as pd
from src.data.io_utils import cargar_dataset, guardar_dataset
from src.utils.print_utils import print_message
import logging

def eliminar_nan_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permite ver las columnas con valores NaN y eliminar los NaN de la columna seleccionada sobre un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a analizar y limpiar.

    Returns:
        pd.DataFrame: DataFrame limpio según la columna seleccionada.
    """
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if nan_cols.empty:
        print("✅ No hay columnas con valores NaN en el DataFrame.")
        return df

    print("\n🔍 Columnas con valores NaN:")
    for i, (col, count) in enumerate(nan_cols.items(), 1):
        print(f"{i}. {col}: {count} NaN")

    try:
        entrada = input("\n👉 Ingresa los números de las columnas (separados por comas) de las que deseas eliminar los NaN (0 para cancelar): ").strip()
        if entrada == "0":
            print("🔙 Operación cancelada.")
            return df

        indices = [int(i.strip()) for i in entrada.split(",") if i.strip().isdigit()]
        if any(i < 1 or i > len(nan_cols) for i in indices):
            print("❌ Uno o más números están fuera del rango válido.")
            return df

        columnas_a_limpiar = [nan_cols.index[i - 1] for i in indices]
        df = df.dropna(subset=columnas_a_limpiar)
        print(f"✅ Se eliminaron los registros con NaN en las columnas: {', '.join(columnas_a_limpiar)}.")
        return df
    except Exception as e:
        print(f"❌ Error al procesar la entrada: {e}")
        return df

    
def eliminar_ceros_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permite eliminar filas con ceros en las columnas seleccionadas, mostrando las columnas con ceros
    y actualizando después de cada eliminación.

    Args:
        df (pd.DataFrame): DataFrame a analizar y limpiar.

    Returns:
        pd.DataFrame: DataFrame limpio según las columnas seleccionadas.
    """

    while True:
        zero_counts = (df == 0).sum()
        zero_cols = zero_counts[zero_counts > 0]

        if zero_cols.empty:
            print("✅ No quedan columnas con valores 0 en el DataFrame.")
            break

        print("\n🔍 Columnas con valores 0:")
        for i, (col, count) in enumerate(zero_cols.items(), 1):
            print(f"{i}. {col}: {count} ceros")

        try:
            entrada = input("\n👉 Ingresa el número de la columna de la que deseas eliminar las filas con ceros (0 para salir): ").strip()
            if entrada == "0":
                print("🔚 Operación finalizada por el usuario.")
                break

            opcion = int(entrada)
            if 1 <= opcion <= len(zero_cols):
                col_a_limpiar = zero_cols.index[opcion - 1]
                df = df[df[col_a_limpiar] != 0]
                print(f"✅ Se eliminaron las filas con ceros en la columna '{col_a_limpiar}'.")
            else:
                print("❌ Opción fuera de rango.")
        except ValueError:
            print("❌ Entrada no válida. Ingresa un número válido.")

    return df

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
        salario_minimo = 828116
    elif año == 2020:
        salario_minimo = 877803
    else:
        salario_minimo = 908526
        
    if año <= 2020:
        # Estatuto viejo (ejemplo)
        if avaluo < 120*salario_minimo:
            return 0.0055
        elif avaluo < 200*salario_minimo:
            return 0.0065
        elif avaluo < 300*salario_minimo:
            return 0.007
        else:
            return 0.0075
    else:
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
    
    # 1. Calcular la moda por barrio (excluyendo 0, 8, 9)
    moda_por_barrio = (
        df[~df['ESTRATO'].isin([0, 8, 9])]
        .groupby('BARRIO')['ESTRATO']
        .agg(lambda x: x.mode()[0])
    )
    
    # 2. Calcular la moda global para barrios no encontrados
    moda_global = df[~df['ESTRATO'].isin([0, 8, 9])]['ESTRATO'].mode()[0]
    
    # 3. Convertir a diccionario para mejor performance
    moda_dict = moda_por_barrio.to_dict()
    
    # 4. Aplicar la corrección
    df['ESTRATO'] = df.apply(
        lambda row: moda_dict.get(row['BARRIO'], moda_global) 
                   if row['ESTRATO'] in [0, 8, 9] 
                   else row['ESTRATO'],
        axis=1
    )
    
    logging.info(f"Estratos corregidos. Moda global: {moda_global}")
    return df

def run(carpeta: str):
    """
    Ejecuta la limpieza de datos en el DataFrame.

    Args:
        carpeta (str): Ruta de la carpeta donde se encuentran los archivos CSV.
    """
    print_message("Limpiando datos")
    # Cargar el DataFrame desde un archivo CSV
    input_path = f"data/processed/{carpeta}/Dataset_transversal.csv"
    output_path = f"data/processed/{carpeta}/Dataset_transversal_depurada.csv"

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