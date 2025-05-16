import seaborn as sns
import matplotlib.pyplot as plt
import os

def generar_grafico_distribucion(df, carpeta, archivo):
    """
    Genera un gráfico de distribución para una columna seleccionada del DataFrame
    y lo guarda como una imagen en formato PNG.

    Args:
        df (pd.DataFrame): DataFrame del cual se generará el gráfico.

    Returns:
        None
    """
    # Crear la carpeta de salida si no existe
    output_dir = f"data/graphs/{carpeta}/{archivo}/"
    os.makedirs(output_dir, exist_ok=True)

    while True:
        try:
            for i, col in enumerate(df.columns, 1):
                print(f"📊 Generando gráfico de distribución para la columna '{col}'...")
                print(f"{i}. {col}")
                plt.figure(figsize=(10, 6))
                sns.histplot(df[col].sample(10000), kde=False, color="blue")
                plt.title(f"Distribución de {col}")
                plt.xlabel(col)
                plt.ylabel("Frecuencia")

                # Guardar el gráfico como imagen
                output_path = os.path.join(output_dir, f"{col}.png")
                plt.savefig(output_path, format="png", dpi=100)
                plt.close()  # Cerrar el gráfico para liberar memoria

                print(f"✅ Gráfico guardado en: {output_path}")

                serie = df[col].dropna()
                print(f"\n📊 Estadísticos de '{col}':")
                print(f"Media: {serie.mean():.4f}")
                print(f"Mediana: {serie.median():.4f}")
                print(f"Desviación estándar: {serie.std():.4f}")
                print(f"Mínimo: {serie.min():.4f}")
                print(f"Máximo: {serie.max():.4f}")
                print(f"Percentil 25: {serie.quantile(0.25):.4f}")
                print(f"Percentil 50: {serie.quantile(0.50):.4f}")
                print(f"Percentil 75: {serie.quantile(0.75):.4f}")

        except ValueError:
            print("❌ Ingresa un número válido.")       
                

        