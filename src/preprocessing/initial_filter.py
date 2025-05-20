from src.data.io_utils import cargar_dataset, guardar_dataset
from utils.data_utils import eliminar_columnas, filtrar
from src.preprocessing.clean_data import eliminar_nan_df
from src.utils.print_utils import print_message
import pandas as pd
from preprocessing.clean_data import corregir_estratos

import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

def run(carpeta: str):
    print_message("Generando base inicial")
    
    input_path = "data/raw/LIQ_IPU_CONS_2024.csv"
    output_path = f"data/processed/{carpeta}/Dataset_panel_depurada.csv"

    pd.set_option("display.float_format", "{:.0f}".format)

    df = cargar_dataset(input_path)

    df = corregir_estratos(df)

    df = eliminar_columnas(df, ['DESTINO_ECONOMICO', 'ESTRATO_SOCIAL', 'TARIFA_PREDIAL', 'FACTCAR','LIQIPU',
                                'MORA', 'DEP_MORA', 'MORA_DEF', 'CANTMORA', 'MORATOT', 'BARRIO'
    ])

    df = filtrar(df, "VIGENCIA", [2022, 2023, 2024])
    
    df = eliminar_nan_df(df)

    df = filtrar(df, "DESTINO_DESCRIPCION", [
        "NULO", "AGROINDUSTRIAL", "SERVICIO FUNERARIO", "AGROFORESTAL", "CULTURAL", "SALUBRIDAD",
        "INFRAESTRUCTURA HIDRICA", "PECUARIO", "FORESTAL", "LOTE NO URBANIZABLE (ServEsp)", "RECREACIONAL",
        "INSTITUCIONAL", "MINERO", "RELIGIOSO", "AGRICOLA", "INDUSTRIAL", "EDUCATIVO", "USO PUBLICO"
    ])

    guardar_dataset(df, output_path)

