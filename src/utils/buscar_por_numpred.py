import pandas as pd

def run(archivo: str, col: str, dato: str):
    """Busca un dato en una columna específica de un archivo CSV."""
    # Cargar el dataset
    df = pd.read_csv(f"data/processed/{archivo}.csv")

    # Filtrar el dataset
    resultado = df[df[col].astype(str).str.contains(dato, na=False)]

    # Mostrar resultado
    if not resultado.empty:
        print("El predio está en el dataset:")
        print(resultado)
    else:
        print("El predio NO está en el dataset.")

if __name__ == "__main__":
    # Ejemplo de uso desde el terminal
    archivo = input("👉 Ingresa el nombre del archivo (sin extensión): ").strip()
    col = input("👉 Ingresa el nombre de la columna: ").strip()
    dato = input("👉 Ingresa el dato a buscar: ").strip()
    run(archivo, col, dato)
