from utils.oi_utils import cargar_dataset, guardar_dataset
from utils.data_utils import eliminar_columnas, eliminar_nulos, filtrar
from colorama import Fore, Style
import pandas as pd

import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

def run(carpeta: str):
    input_path = "data/raw/LIQ_IPU_CONS_2024.csv"
    output_path = f"data/processed/{carpeta}/1_Dataset_panel_depurada.csv"

    pd.set_option("display.float_format", "{:.0f}".format)

    df = cargar_dataset(input_path)

    df = eliminar_columnas(df, ['MORATOT', 'CANTMORA', 'MORA_DEF', 'DEP_MORA', 'MORA', 'FACTCAR',
        'ESTRATO_SOCIAL', 'LIQIPU', 'BARRIO'
    ])

    df = filtrar(df, "VIGENCIA", [2023, 2024])
    df = eliminar_nulos(df, 'NUMPRED')

    df = filtrar(df, "DESTINO_DESCRIPCION", [
        "NULO", "AGROINDUSTRIAL", "SERVICIO FUNERARIO", "AGROFORESTAL", "CULTURAL", "SALUBRIDAD",
        "INFRAESTRUCTURA HIDRICA", "PECUARIO", "FORESTAL", "LOTE NO URBANIZABLE (ServEsp)", "RECREACIONAL",
        "INSTITUCIONAL", "MINERO", "RELIGIOSO", "AGRICOLA", "INDUSTRIAL", "EDUCATIVO", "USO PUBLICO"
    ])

    guardar_dataset(df, output_path)

    num_unicos = df["NUMPRED"].nunique()
    logging.info(f"{Fore.GREEN}🔢 Cantidad de NUMPRED únicos: {num_unicos}{Style.RESET_ALL}")
