import joblib
from sklearn.tree import export_text
import pandas as pd

def mostrar_estructura_modelo(ruta_modelo, nombres_variables):
    """
    Muestra la estructura básica del modelo guardado en joblib
    
    Parámetros:
        ruta_modelo: str - Ruta al archivo .joblib
        nombres_variables: list - Nombres de las variables de entrada
    """

    modelo = joblib.load(f"src/models/{ruta_modelo}")
    
    print(f"\n🔍 Estructura del modelo: {type(modelo).__name__}")
    print("="*50)
    
    if hasattr(modelo, 'tree_'):  # Árbol de decisión
        print("🌳 Estructura del árbol:")
        print(export_text(modelo, feature_names=nombres_variables))
        
    elif hasattr(modelo, 'coef_'):  # Regresión logística
        print("📈 Coeficientes de regresión:")
        coef_df = pd.DataFrame({
            'Variable': nombres_variables,
            'Coeficiente': modelo.coef_[0]
        })
        print(coef_df.to_string(index=False))
        
    else:  # Red neuronal u otros
        print("🕵️ Este modelo no tiene una estructura imprimible directamente")
        print("Tipo de modelo:", type(modelo).__name__)

def run():
    """
    Función principal para analizar la estructura de los modelos entrenados.
    """
    print("🔍 Análisis de la estructura de los modelos entrenados")
    print("="*50)
    
    # Variables utilizadas en los modelos
    variables = ['VARIACION_AVALUO', 'VARIACION_TARIFA', 'DESCUENTO', 
             'AREA_CONSTRUIDA', 'ANIOS_PAGADOS', 'ESTRATO']

    # Mostrar estructura de cada modelo
    mostrar_estructura_modelo('Arbol_urbano.joblib', variables)
    mostrar_estructura_modelo('MLP_urbano.joblib', variables)
    mostrar_estructura_modelo('RL_urbano.joblib', variables)