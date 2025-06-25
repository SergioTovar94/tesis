import os
from src import config
import pandas as pd
from colorama import Fore, Style

def cargar_dataset(ruta: str) -> pd.DataFrame:
    """Carga un dataset desde la ruta especificada."""
    if not os.path.exists(ruta):
        print(f"{Fore.RED}❌ Error: El archivo {ruta} no existe.{Style.RESET_ALL}")
        return None
    df = pd.read_csv(ruta, dtype={'columna_0': str})
    print(f"{Fore.GREEN}✅ Dataset cargado desde {ruta}{Style.RESET_ALL}")
    return df

def guardar_dataset(df: pd.DataFrame, ruta: str):
    """Guarda un dataset en la ruta especificada, asegurando que el directorio exista."""
    if config.VARIABLE_OBJETIVO in df.columns:
        columna_pago = df.pop(config.VARIABLE_OBJETIVO)  # Quita la columna
        df[config.VARIABLE_OBJETIVO] = columna_pago
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    df.to_csv(ruta, index=False)
    print(f"{Fore.CYAN}💾 Dataset guardado en {ruta}{Style.RESET_ALL}")

def ubicar_en_raiz():
    import os
    import shutil

    carpeta_base = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO)
    # Asegurarse de que la carpeta destino exista
    os.makedirs(config.CARPETA_LOCAL, exist_ok=True)

    # Recorrer todos los archivos en la carpeta origen
    for nombre_archivo in os.listdir(carpeta_base):
        ruta_archivo_origen = os.path.join(carpeta_base, nombre_archivo)

        # Verificar si es un archivo (no una carpeta)
        if os.path.isfile(ruta_archivo_origen):
            shutil.copy(ruta_archivo_origen, config.CARPETA_LOCAL)

    print("Archivos copiados exitosamente.")