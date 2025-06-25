import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.utils.data_utils import eliminar_columnas
from src import config

def run(df, carpeta, archivo):
    """
    Genera un gráfico de distribución para una columna seleccionada del DataFrame
    y lo guarda como una imagen en formato PNG.
    """
    # Crear la carpeta de salida si no existe
    nombre_sin_extension, _ = os.path.splitext(archivo)
    output_dir = os.path.join(config.DATA_GRAPHS, 'Distribuciones', nombre_sin_extension)
    os.makedirs(output_dir, exist_ok=True)

    # Eliminar columnas no numéricas
    df = eliminar_columnas(df, [config.LLAVE])
    df_num = df.select_dtypes(include='number')

    while True:
        try:
            for i, col in enumerate(df_num.columns, 1):
                print(f"📊 Generando gráfico de distribución para la columna '{col}'...")
                print(f"{i}. {col}")
                plt.figure(figsize=(10, 6))
                sns.histplot(df_num[col].sample(10000), kde=False, color="blue")
                plt.title(f"Distribución de {col}")
                plt.xlabel(col)
                plt.ylabel("Frecuencia")

                # Guardar el gráfico como imagen
                output_path = os.path.join(output_dir, f"{col}.png")
                plt.savefig(output_path, format="png", dpi=100)
                plt.close()  # Cerrar el gráfico para liberar memoria

                print(f"✅ Gráfico guardado en: {output_path}")

                serie = df_num[col].dropna()
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
                

        