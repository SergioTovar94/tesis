import os
from src import config
from eda.matriz_correlacion import calcular_matriz_correlacion
from eda.vif import calcular_vif
from eda.distribucion import generar_grafico_distribucion
from tools.menu import seleccionar_opcion
from src.data.io_utils import cargar_dataset
from utils.data_utils import eliminar_columnas


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
                calcular_matriz_correlacion(df, graficar=True)

        elif opcion == "Calcular VIF (Factor de Inflación de Varianza)":
            
            if df is not None:
                vif = calcular_vif(df)
                print("\n🔍 VIF calculado:")
                print(vif)

        elif opcion == "Generar gráfico de distribución":
            if df is not None:
                generar_grafico_distribucion(df, config.TIPO_VARIABLE_OBJETIVO, archivo)

        elif opcion is None:
            print("🔙 Volviendo al menú principal.")
            break