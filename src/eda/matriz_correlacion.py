import seaborn as sns
import matplotlib.pyplot as plt

def calcular_matriz_correlacion(df, metodo="pearson", graficar=False):
    """
    Calcula la matriz de correlación de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        metodo (str): Método de correlación ('pearson', 'spearman', 'kendall').
        graficar (bool): Si es True, genera un heatmap de la matriz de correlación.

    Returns:
        pd.DataFrame: Matriz de correlación.
    """
    correlacion = df.corr(method=metodo)
    if graficar:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlacion, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matriz de Correlación")
        plt.show()
    return correlacion