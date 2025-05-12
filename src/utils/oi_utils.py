import os
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
    if 'COMPORTAMIENTO_PAGO' in df.columns:
        columna_pago = df.pop('COMPORTAMIENTO_PAGO')  # Quita la columna
        df['COMPORTAMIENTO_PAGO'] = columna_pago
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    df.to_csv(ruta, index=False)
    print(f"{Fore.CYAN}💾 Dataset guardado en {ruta}{Style.RESET_ALL}")

def actualizar_para_weka():
    import os
    import shutil

    # Ruta de la carpeta de origen
    carpeta_origen = 'data/processed/com_si_no/'

    # Ruta de la carpeta de destino
    carpeta_destino = 'C:\\Users\\sergi\\OneDrive'

    # Asegurarse de que la carpeta destino exista
    os.makedirs(carpeta_destino, exist_ok=True)

    # Recorrer todos los archivos en la carpeta origen
    for nombre_archivo in os.listdir(carpeta_origen):
        ruta_archivo_origen = os.path.join(carpeta_origen, nombre_archivo)

        # Verificar si es un archivo (no una carpeta)
        if os.path.isfile(ruta_archivo_origen):
            shutil.copy(ruta_archivo_origen, carpeta_destino)

    print("Archivos copiados exitosamente.")