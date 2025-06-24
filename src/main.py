from src import config
from tools.menu import seleccionar_opcion, seleccionar_archivo

from src.preprocessing.pipeline import alistar_datasets
from features.pipeline import transformar_dataset
from statistic import analizar_dataset
from models.modelos import run as entrenar_modelos
from models.arboles import run as entrenar_arboles
from models.analizar_modelo import run as analizar_modelos

def main():
    """Función principal para elegir entre alistar datasets o entrenar modelos."""
    while True:
        print("\n📌 Opciones disponibles:")
        print("1️⃣ Alistar datasets")
        print("2️⃣ Transformar dataset")
        print("3️⃣ Análisis de datasets")
        print("4️⃣ Entrenar modelos")
        print("5️⃣ Validar modelo existente")
        print("6️⃣ Probar árboles")
        print("0️⃣ Salir")

        opcion = input("\n👉 Selecciona una opción: ").strip()

        if opcion == "1":
            alistar_datasets()
        elif opcion == "2":
            archivo = seleccionar_archivo(config.TIPO_VARIABLE_OBJETIVO)
            transformar_dataset(archivo)
        elif opcion == "3":
            archivo = seleccionar_archivo(config.TIPO_VARIABLE_OBJETIVO)
            analizar_dataset(archivo)
        elif opcion == "4":
            archivo = seleccionar_archivo(config.TIPO_VARIABLE_OBJETIVO)
            entrenar_modelos(archivo)
        elif opcion == "5":
            analizar_modelos()
        elif opcion == "6":
            entrenar_arboles(config.TIPO_VARIABLE_OBJETIVO)
        elif opcion == "0":
            print("👋 Finalizando el programa.")
            break

        else:
            print("❌ Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()