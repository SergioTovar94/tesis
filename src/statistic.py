
from eda.matriz_correlacion import calcular_matriz_correlacion
from eda.vif import calcular_vif
from eda.box_tidwell import calcular_box_tidwell
from eda.distribucion import generar_grafico_distribucion
from tools.menu import seleccionar_opcion
from src.data.io_utils import cargar_dataset
from utils.data_utils import eliminar_columnas


def analizar_dataset(carpeta, archivo):
    """Permite realizar análisis sobre un dataset seleccionado."""
    menu_analisis = {
        "1": "Calcular matriz de correlación",
        "2": "Calcular VIF (Factor de Inflación de Varianza)",
        "3": "Generar gráfico de distribución"
    }
    ruta = f"data/processed/{carpeta}/{archivo}"
    df = cargar_dataset(ruta)

    while True:
        print("\n🔍 Columnas disponibles en el dataset:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")

        print("\nSi deseas eliminar una columna, ingresa su número.")
        print("Si no deseas eliminar más columnas, ingresa 0.")

        try:
            opcion = int(input("\n👉 Selecciona una opción: ").strip())
            if opcion == 0:
                print("✅ Finalizando la eliminación de columnas.")
                break
            elif 1 <= opcion <= len(df.columns):
                col_a_eliminar = df.columns[opcion - 1]
                df = eliminar_columnas(df, [col_a_eliminar])
                print(f"✅ Columna '{col_a_eliminar}' eliminada.")
            else:
                print("❌ Opción no válida. Intenta de nuevo.")
        except ValueError:
            print("❌ Ingresa un número válido.")

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
                generar_grafico_distribucion(df, carpeta, archivo)

        elif opcion is None:
            print("🔙 Volviendo al menú principal.")
            break