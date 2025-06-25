import seaborn as sns
import matplotlib.pyplot as plt
import os
from src import config
import pandas as pd

def run(df: pd.DataFrame, nombre_archivo: str):
    """
    Calcula la matriz de correlación de las variables numéricas de un DataFrame.
    """
    df_num = df.select_dtypes(include='number')
    correlacion = df_num.corr(method='pearson')
    tam  = len(df.columns)/2
    plt.figure(figsize=(tam, tam))
    sns.heatmap(correlacion, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlación")
    
    nombre_sin_extension, _ = os.path.splitext(nombre_archivo)
    ruta_salida = os.path.join(config.DATA_GRAPHS, config.MATRICES, nombre_sin_extension)
    
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    plt.savefig(ruta_salida)
    print(f"✅ Heatmap guardado en: {ruta_salida}")
    plt.close()
