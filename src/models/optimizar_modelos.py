import os
import config
import joblib
from src.models.optimizer import optimizar_regresion_logistica, optimizar_arbol_decision, optimizar_mlp

from src.utils.io_utils import cargar_dataset
from utils.modelo_utils import preprocess_data, seleccionar_columnas
from src.utils.menu import seleccionar_opcion

def run(archivo: str):
    try:
        input_path = os.path.join(config.DATA_PROCESSED, config.TIPO_VARIABLE_OBJETIVO, archivo)
        df = cargar_dataset(input_path)
        X_train, X_test, y_train, y_test = preprocess_data(df)
        X_train, X_test = seleccionar_columnas(X_train, X_test) 

        opcion = seleccionar_opcion(config.MENU_MODELOS, "Selecciona el modelo a optimizar")
        print(f"\nHas seleccionado la opción: {opcion}")
        resultados = {}
            
        if opcion in ['Regresión Logística', 'Optimizar los 3 modelos']:
            print("\nOPTIMIZANDO REGRESIÓN LOGÍSTICA...")
            resultados['RL'] = optimizar_regresion_logistica(X_train, y_train, X_test, y_test)
        if opcion in ['Árbol de Decisión', 'Optimizar los 3 modelos']:
            print("\nOPTIMIZANDO ÁRBOL DE DECISIÓN...")
            resultados['Arbol'] = optimizar_arbol_decision(X_train, y_train, X_test, y_test)    
        if opcion in ['Perceptrón Multicapa (MLP)', 'Optimizar los 3 modelos']:
            print("\nOPTIMIZANDO PERCEPTRÓN MULTICAPA...")
            resultados['MLP'] = optimizar_mlp(X_train, y_train, X_test, y_test)

        modelos_guardados = {}

        if resultados:
            print("\nRESULTADOS DE OPTIMIZACIÓN:")
            print("-" * 70)
            print(f"{'Modelo':<20} | {'Mejores Parámetros':<30} | {'F1-Score (NO)':<12} | {'Exactitud':<10}")
            print("-" * 70)
            for modelo, datos in resultados.items():
                print(f"{modelo:<20}|{str(datos['mejores_params']):<30}|{datos['f1_no']:.4f}|{datos['exactitud']:.4f}")                
                # Guardar el modelo
                ruta_modelo = os.path.join(config.CARPETA_MODELOS, config.TIPO_VARIABLE_OBJETIVO, f'{modelo}.joblib')
                os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
                joblib.dump({
                    'modelo': datos['mejor_modelo'],
                    'feature_names': X_train.columns.tolist()  
                    }, ruta_modelo)
                modelos_guardados[modelo] = ruta_modelo
                print(f"✅ Modelo guardado en: {ruta_modelo}")
    except Exception as e:
        print(f"❌ Error al procesar la entrada: {e}")