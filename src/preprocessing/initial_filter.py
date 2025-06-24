import os
from src import config
from src.data.io_utils import cargar_dataset, guardar_dataset
from utils.data_utils import eliminar_columnas, filtrar
from src.preprocessing.clean_data import eliminar_nan_df
from src.utils.print_utils import print_message
import pandas as pd
from preprocessing.clean_data import corregir_estratos

import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

def run():
    print_message("Generando base inicial")

    input_path = config.DATA_ORIGINAL
    output_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, config.PANEL_DEPURADA)

    pd.set_option("display.float_format", "{:.0f}".format)

    df = cargar_dataset(input_path)

    df = corregir_estratos(df)

    df = eliminar_columnas(df, config.COLUMNAS_A_ELIMINAR)

    df = filtrar(df, "VIGENCIA", config.ANIOS_A_FILTRAR)
    
    df = eliminar_nan_df(df)

    df = filtrar(df, "DESTINO_DESCRIPCION", config.DESTINOS_A_FILTRAR)

    guardar_dataset(df, output_path)

