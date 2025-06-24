import joblib
from sklearn.tree import export_text, plot_tree
import pandas as pd
import matplotlib.pyplot as plt

def mostrar_estructura_modelo(ruta_modelo):
    """
    Muestra la estructura básica del modelo guardado en joblib y evalúa su interpretabilidad.
    """

    obj = joblib.load(f"src/models/{ruta_modelo}")
    modelo = obj['modelo']
    nombres_variables = obj['feature_names']
    tipo = type(modelo).__name__

    print(f"\n🔍 Estructura del modelo: {tipo}")
    print("=" * 60)

    if hasattr(modelo, 'tree_'):

        # Mostrar árbol
        num_nodos = modelo.tree_.node_count
        alto = max(8, int(num_nodos / 2)) 
        plt.figure(figsize=(alto*1.5, alto))
        plot_tree(modelo,
                  feature_names=nombres_variables,
                  class_names=True,
                  filled=True,
                  rounded=True,
                  fontsize=10)
        plt.savefig("data/graphs/arbol/arbol_decision.png")
        print("📊 Imagen del árbol guardada en: data/graphs/arbol/arbol_decision.png")
        plt.close()

        # Texto del árbol
        print("🌳 Reglas del árbol:")
        texto = export_text(modelo, feature_names=nombres_variables, show_weights=True)
        print(texto)

        # Ejemplo lenguaje natural
        print("\n📖 Ejemplo de explicación en lenguaje natural:")
        for linea in texto.split("\n")[:3]:
            regla = linea.strip().replace("|--- ", "")
            print(f"- Si {regla}, entonces se sigue esta rama del árbol")

    elif hasattr(modelo, 'coef_'):
        print("📈 Coeficientes de la regresión logística:")
        coef_df = pd.DataFrame({
            'Variable': nombres_variables,
            'Coeficiente': modelo.coef_[0]
        })
        print(coef_df.to_string(index=False))

    elif hasattr(modelo, 'coefs_'):
        print("🤖 Red neuronal multicapa (MLP):")
        print("Este modelo no tiene una estructura directamente interpretable.")
        # No actualizamos ningún criterio, sigue siendo caja negra.

    else:
        print("⚠️ Modelo no reconocido para interpretación.")

def run():
    """
    Función principal para analizar la estructura de los modelos entrenados.
    """
    print("🔍 Análisis de la estructura de los modelos entrenados")
    print("=" * 60)

    mostrar_estructura_modelo('Arbol_urbano.joblib')
    mostrar_estructura_modelo('MLP_urbano.joblib')
    mostrar_estructura_modelo('RL_urbano.joblib')
