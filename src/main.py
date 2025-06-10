from src.preprocessing.pipeline import alistar_datasets
from models.modelos import run as entrenar_modelos
from tools.menu import seleccionar_opcion, seleccionar_archivo
from statistic import analizar_dataset
from features.pipeline import transformar_dataset

CARPETAS_DATASETS = {
    "1": "com_dejo_pagar",
    "2": "com_si_no"
}

ESTADISTICOS = {
    "1": "Calcular matriz de correlación",
    "2": "Calcular VIF (Factor de Inflación de Varianza)",
}

ZONAS_DISPONIBLES = {
    "1": "urbano",
    "2": "rural",
}

def main():
    """Función principal para elegir entre alistar datasets o entrenar modelos."""
    while True:
        print("\n📌 Opciones disponibles:")
        print("1️⃣ Alistar datasets")
        print("2️⃣ Transformar dataset")
        print("3️⃣ Análisis de datasets")
        print("4️⃣ Entrenar modelos")
        print("0️⃣ Salir")

        opcion = input("\n👉 Selecciona una opción: ").strip()

        if opcion == "1":
            carpeta = seleccionar_opcion(CARPETAS_DATASETS, "Carpetas disponibles")
            if carpeta:
                alistar_datasets(carpeta)
        elif opcion == "2":
            carpeta = seleccionar_opcion(CARPETAS_DATASETS, "Selecciona un estadistico")
            archivo = seleccionar_archivo(carpeta)
            if carpeta:
                transformar_dataset(carpeta, archivo)
        elif opcion == "3":
            carpeta = seleccionar_opcion(CARPETAS_DATASETS, "Selecciona un estadistico")
            archivo = seleccionar_archivo(carpeta)
            if carpeta:
                analizar_dataset(carpeta, archivo)
        elif opcion == "4":
            carpeta = seleccionar_opcion(CARPETAS_DATASETS, "Selecciona dataset")
            if carpeta:
                zona = seleccionar_opcion(ZONAS_DISPONIBLES, "Zonas disponibles")
                if zona:
                    entrenar_modelos(carpeta, zona)
        elif opcion == "0":
            print("👋 Finalizando el programa.")
            break

        else:
            print("❌ Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()