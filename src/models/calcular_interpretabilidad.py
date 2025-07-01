import os
import numpy as np
import joblib
from src import config

def get_paths_depths(decision_tree):
    tree = decision_tree.tree_
    children_left = tree.children_left
    children_right = tree.children_right

    leaf_depths = []

    def traverse(node, current_depth):
        if children_left[node] == children_right[node]:  # Hoja
            leaf_depths.append(current_depth)
        else:
            if children_left[node] != -1:
                traverse(children_left[node], current_depth + 1)
            if children_right[node] != -1:
                traverse(children_right[node], current_depth + 1)

    traverse(0, 0)  # Empieza en la raíz
    return leaf_depths

def calcular(ruta_modelo):
    obj = joblib.load(ruta_modelo)
    modelo = obj['modelo']
    
    if hasattr(modelo, 'tree_'):
        operaciones = np.mean(get_paths_depths(modelo))
        componentes_interpretables = modelo.tree_.node_count
        componentes_totales = modelo.tree_.node_count
        profundidad = modelo.get_depth()
    elif hasattr(modelo, 'coefs_'):
        hidden_layer_sizes = modelo.hidden_layer_sizes
        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = (hidden_layer_sizes,)
        n_inputs = modelo.n_features_in_
        layers = list(hidden_layer_sizes) + [1] 
        operaciones = 0
        inputs = n_inputs
        n_inputs = modelo.n_features_in_
        for neurons in layers:
            ops = neurons * inputs
            operaciones += ops
            inputs = neurons

        componentes_interpretables = 0
        neuronas_ocultas = sum(modelo.hidden_layer_sizes)
        neuronas_salida = modelo.n_outputs_
        componentes_totales = neuronas_ocultas + neuronas_salida
        profundidad = componentes_totales
    elif hasattr(modelo, 'coef_'):
        p = modelo.coef_.shape[1]
        operaciones = 2 * p + 3
        componentes_interpretables = p
        componentes_totales = p + 1
        profundidad = componentes_totales

    print("Cantidad de operaciones o decisiones que el modelo debe realizar para llegar a una predicción:", operaciones)
    print("Componentes interpretables: ", componentes_interpretables)
    print("Componentes totales: ", componentes_totales)
    print(f"D: Profundidad máxima (para árboles de decisión) P: Número de parámetros para redes y regresión:", profundidad)



def run():
    carpeta_modelos = os.path.join(config.CARPETA_MODELOS, config.TIPO_VARIABLE_OBJETIVO)
    modelos_disponibles = [
                f for f in os.listdir(carpeta_modelos) if f.endswith('.joblib')
            ]
    if not modelos_disponibles:
        print("❌ No hay modelos guardados.")
    
    print("\n📂 Modelos disponibles:")
    for idx, modelo in enumerate(modelos_disponibles, 1):
        print(f"{idx} - {modelo}")
    try:
        opcion_modelo = int(input("\nSelecciona un modelo (número): "))
        if 1 <= opcion_modelo <= len(modelos_disponibles):
            modelo_elegido = modelos_disponibles[opcion_modelo - 1]
            ruta_modelo = os.path.join(carpeta_modelos, modelo_elegido)
            calcular(ruta_modelo)
        else:
            print("❌ Opción fuera de rango.")
    except ValueError:
        print("❌ Opción inválida. Debe ser un número.")
