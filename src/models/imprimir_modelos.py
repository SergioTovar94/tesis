import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text

from src import config

def mostrar_estructura_modelo(ruta_modelo):
    """
    Muestra la estructura básica del modelo guardado en joblib y evalúa su interpretabilidad.
    """
    obj = joblib.load(ruta_modelo)
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
        output_dir = os.path.join(config.DATA_GRAPHS, config.ARBOL, "arbol_decision.png")
        plt.savefig(output_dir)
        print(f"Imagen del árbol guardada en: {output_dir}")
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
    carpeta_modelos = os.path.join(config.CARPETA_MODELOS, config.TIPO_VARIABLE_OBJETIVO)
    modelos_disponibles = [
                f for f in os.listdir(carpeta_modelos) if f.endswith('.joblib')
            ]
    if not modelos_disponibles:
        print("❌ No hay modelos guardados.")
    
    print("\n📂 Modelos disponibles para imprimir:")
    for idx, modelo in enumerate(modelos_disponibles, 1):
        print(f"{idx} - {modelo}")
    try:
        opcion_modelo = int(input("\nSelecciona un modelo (número): "))
        if 1 <= opcion_modelo <= len(modelos_disponibles):
            modelo_elegido = modelos_disponibles[opcion_modelo - 1]
            ruta_modelo = os.path.join(carpeta_modelos, modelo_elegido)

            print(f"\n✅ Imprimiendo modelo: {modelo_elegido}")
            mostrar_estructura_modelo(ruta_modelo)
            print("✅ Evaluación completada.")
        else:
            print("❌ Opción fuera de rango.")
    except ValueError:
        print("❌ Opción inválida. Debe ser un número.")