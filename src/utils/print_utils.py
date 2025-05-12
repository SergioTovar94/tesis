import pandas as pd
from colorama import Fore, Style

def imprimir_nulos(df: pd.DataFrame):
    nulos_por_columna = df.isnull().sum()
    print(nulos_por_columna[nulos_por_columna > 0])

def print_message(step_number, description, color=Fore.LIGHTYELLOW_EX):
    print(f"{color}Ejecutando script{step_number}: {description}...{Style.RESET_ALL}")