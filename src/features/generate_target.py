import os
import pandas as pd
from src import config
from src.utils.io_utils import cargar_dataset, guardar_dataset
from src.utils.data_utils import eliminar_columnas, filtrar_por_anio
from src.preprocessing.clean_data import eliminar_nan_df
from src.utils.print_utils import print_message

def separarDataset(df: pd.DataFrame, anio: int)-> pd.DataFrame:
    if anio == 2019:
        df = df[df["TIPO_PREDIO"] == 1]
    else:
        df = df[df["TIPO_PREDIO"] == 0]
    df = eliminar_columnas(df, ['TIPO_PREDIO'])
    return df

def generar_comportamiento(df: pd.DataFrame, forma: str, anio: int) -> pd.DataFrame:
    if forma == 'com_si_no':
        df['COMPORTAMIENTO_PAGO'] = df.apply(
            lambda row: 'SI' if (row[f'RECIPU_{anio + 2}'] > 0)
            else 'NO',
            axis=1
        )
    else:
        df['COMPORTAMIENTO_PAGO'] = df.apply(
            lambda row: 'Siempre Paga' if (
                        row[f'RECIPU_{anio}'] > 0 and row[f'RECIPU_{anio + 1}'] > 0 and row[f'RECIPU_{anio + 2}'] > 0)
            else 'Dejo de Pagar' if (
                        row[f'RECIPU_{anio}'] > 0 and row[f'RECIPU_{anio + 1}'] > 0 and row[f'RECIPU_{anio + 2}'] == 0)
            else 'Nunca paga' if (row[f'RECIPU_{anio}'] == 0 and row[f'RECIPU_{anio + 1}'] == 0 and row[
                f'RECIPU_{anio + 2}'] == 0)
            else 'Otro',
            axis=1
        )
    return df

def run():
    print_message(f"Generando comportamiento de pago de {config.ZONA_POR_DEFECTO}")
    input_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, config.TRANSVERSAL_DEPURADA)
    output_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, config.URBANO)

    df_inicio = cargar_dataset(input_path)

    df = eliminar_nan_df(df_inicio)

    if config.ZONA_POR_DEFECTO == 'urbano':
        df = filtrar_por_anio(df, config.ANIO_URBANO)
    elif config.ZONA_POR_DEFECTO == 'rural':
        df = filtrar_por_anio(df, config.ANIO_RURAL)

    # Crear la columna 'Comportamiento_Pago'
    df = generar_comportamiento(df, config.TIPO_VARIABLE_OBJETIVO, config.ANIO_OBJETIVO)

    #Crear la columna de cuantos anios ha pagado el propietario
    df["ANIOS_PAGADOS"] = (df[f'RECIPU_{config.ANIO_OBJETIVO}'] > 0).astype(int) + \
                         (df[f'RECIPU_{config.ANIO_OBJETIVO+1}'] > 0).astype(int)

    df = separarDataset(df, config.ANIO_OBJETIVO)

    # Guardar en archivos separado
    guardar_dataset(df, output_path)
