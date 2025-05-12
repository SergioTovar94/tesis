from src.utils.oi_utils import cargar_dataset, guardar_dataset
from scipy.stats import zscore

def run(dataset: str, carpeta: str):

    input_path = f"data/processed/{carpeta}/{dataset}.csv"
    output_path = f"data/processed/{carpeta}/{dataset}_estandarizado.csv"

    df = cargar_dataset(input_path)

    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    df[columnas_numericas] = df[columnas_numericas].apply(zscore)

    guardar_dataset(df, output_path)
