import os
from src import config
from src.utils.io_utils import cargar_dataset, guardar_dataset
from utils.data_utils import eliminar_columnas, filtrar
from src.preprocessing.clean_data import eliminar_nan_df, corregir_estratos
from src.utils.print_utils import print_message

def run():
    """
    Ejecuta el filtrado inicial y limpieza del dataset:
    - Carga el dataset original.
    - Corrige valores de estrato.
    - Elimina columnas innecesarias.
    - Filtra por años de vigencia.
    - Elimina filas con valores nulos.
    - Filtra por destino.
    - Guarda el dataset limpio en carpeta procesada.
    """
    
    print_message("Generando base inicial")

    input_path = os.path.join(config.DATA_RAW, config.DATA_ORIGINAL)
    output_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, config.PANEL_DEPURADA)

    df = cargar_dataset(input_path)

    df = corregir_estratos(df)

    df = eliminar_columnas(df, config.COLUMNAS_A_ELIMINAR)

    df = filtrar(df, "VIGENCIA", config.ANIOS_A_FILTRAR)
    
    df = eliminar_nan_df(df)

    df = filtrar(df, "DESTINO_DESCRIPCION", config.DESTINOS_A_FILTRAR)

    guardar_dataset(df, output_path)

