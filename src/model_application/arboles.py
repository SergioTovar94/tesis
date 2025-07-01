import os
from src import config
from src.utils.io_utils import cargar_dataset
from utils.modelo_utils import preprocess_data
from src.models.optimizer import optimizar_arbol_decision
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def run():
    try:
        for nombre in sorted(os.listdir(config.RUTA_ARBOLES)):
            if not nombre.startswith("Entrenamiento"):
                continue
            ruta = os.path.join(config.RUTA_ARBOLES, nombre)
            df = cargar_dataset(ruta)
            X_train, X_test, y_train, y_test = preprocess_data(df)
            columnas_existentes = [col for col in config.COLUMNAS_DEF if col in X_train.columns]
            X_train = X_train[columnas_existentes]
            X_test = X_test[columnas_existentes] 

            resultados = {}
                
            print("\nOPTIMIZANDO ÁRBOL DE DECISIÓN...")
            resultados['Arbol'] = optimizar_arbol_decision(X_train, y_train, X_test, y_test)    
            
            modelos_guardados = {}

            if resultados:
                print("\nRESULTADOS DE OPTIMIZACIÓN:")
                print("-" * 70)
                print(f"{'Modelo':<20} | {'Mejores Parámetros':<30} | {'F1-Score (NO)':<12} | {'Exactitud':<10}")
                print("-" * 70)
                for modelo, datos in resultados.items():
                    arbol = datos['mejor_modelo']
                    print(f"{modelo:<20}|{str(datos['mejores_params']):<30}|{datos['f1_no']:.4f}|{datos['exactitud']:.4f}")                
                    # Guardar el modelo
                    nombre_base = nombre.replace("Entrenamiento_dataset_", "").replace(".csv", "")
                    ruta_modelo = os.path.join(config.CARPETA_MODELOS, config.TIPO_VARIABLE_OBJETIVO, f'{nombre_base}_{modelo}.joblib')
                    os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
                    joblib.dump({
                        'modelo': datos['mejor_modelo'],
                        'feature_names': X_train.columns.tolist()  
                        }, ruta_modelo)
                    modelos_guardados[modelo] = ruta_modelo
                    print(f"✅ Modelo guardado en: {ruta_modelo}")
                    num_nodos = arbol.tree_.node_count
                    alto = max(8, int(num_nodos / 2)) 
                    plt.figure(figsize=(alto*1.5, alto))
                    plot_tree(arbol,
                            feature_names=columnas_existentes,
                            class_names=True,
                            filled=True,
                            rounded=True,
                            fontsize=10)
                    output_dir = os.path.join(config.DATA_GRAPHS, config.ARBOL, f'{nombre_base}_{modelo}.png')
                    plt.savefig(output_dir)
                    print(f"Imagen del árbol guardada en: {output_dir}")
                    plt.close()
    except Exception as e:
        print(f"❌ Error al procesar la entrada: {e}")