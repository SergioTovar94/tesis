import os
from src import config
from src.eda import matriz_correlacion, distribucion, vif
from src.utils.menu import seleccionar_opcion
from src.utils.io_utils import cargar_dataset

def analizar_dataset(archivo):
    """Permite realizar análisis sobre un dataset seleccionado."""
    menu_analisis = {
        "1": "Calcular matriz de correlación",
        "2": "Calcular VIF (Factor de Inflación de Varianza)",
        "3": "Generar gráfico de distribución"
    }

    input_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, archivo)
    
    df = cargar_dataset(input_path)

    while True:
        opcion = seleccionar_opcion(menu_analisis, "Análisis disponibles")
        if opcion == "Calcular matriz de correlación":
            
            if df is not None:
                matriz_correlacion.run(df, archivo)

        elif opcion == "Calcular VIF (Factor de Inflación de Varianza)":
            
            if df is not None:
                res_vif = vif.run(df)
                print("\n🔍 VIF calculado:")
                print(res_vif)

        elif opcion == "Generar gráfico de distribución":
            if df is not None:
                distribucion.run(df, config.TIPO_VARIABLE_OBJETIVO, archivo)

        elif opcion is None:
            print("🔙 Volviendo al menú principal.")
            break