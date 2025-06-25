from src import config
from src.utils.menu import seleccionar_archivo
from src.preprocessing.pipeline import alistar_datasets
from src.features.pipeline import transformar_dataset
from src.eda.pipeline import analizar_dataset
from src.models.pipeline import entrenar_modelos
#from src.model_application.pipeline import entrenar_arboles

def opcion_transformar():
    archivo = seleccionar_archivo(config.TIPO_VARIABLE_OBJETIVO)
    transformar_dataset(archivo)

def opcion_analizar():
    archivo = seleccionar_archivo(config.TIPO_VARIABLE_OBJETIVO)
    analizar_dataset(archivo)

def opcion_entrenar():
    archivo = seleccionar_archivo(config.TIPO_VARIABLE_OBJETIVO)
    entrenar_modelos(archivo)

opciones = {
    "1": ("Alistar datasets", alistar_datasets),
    "2": ("Transformar dataset", opcion_transformar),
    "3": ("Análisis de datasets", opcion_analizar),
    "4": ("Entrenar modelos", opcion_entrenar),
    #"5": ("Probar árboles", entrenar_arboles),
    "0": ("Salir", None)
}

def main():
    while True:
        print("\n📌 Opciones disponibles:")
        for k, (desc, _) in opciones.items():
            print(f"{k}️⃣ {desc}")
        opcion = input("\n👉 Selecciona una opción: ").strip()

        if opcion == "0":
            print("👋 Finalizando el programa.")
            break
        elif opcion in opciones:
            _, funcion = opciones[opcion]
            if funcion:
                funcion()
        else:
            print("❌ Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
