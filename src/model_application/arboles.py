from utils.modelo_utils import preprocess_data, seleccionar_columnas
from src.utils.io_utils import cargar_dataset
from sklearn.tree import DecisionTreeClassifier
import os
from src.utils.menu import seleccionar_archivo
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, make_scorer, roc_auc_score,confusion_matrix
import joblib
from datetime import datetime

f1_no_scorer = make_scorer(f1_score, pos_label=1) 

def run(carpeta: str):
    try:
        archivo = seleccionar_archivo(carpeta)
        input_path = f"data/processed/{carpeta}/{archivo}"
        
        if not os.path.exists(input_path):
            print(f"❌ El archivo '{input_path}' no existe. Verifica la ruta.")
            return
        
        try:
            df = cargar_dataset(input_path)
        except Exception as e:
            print(f"❌ Error al cargar el dataset: {e}")
            return
                
        try:
            X_train, X_test, y_train, y_test = preprocess_data(df)
            X_train, X_test = seleccionar_columnas(X_train, X_test)
        except Exception as e:
            print(f"❌ Error en el preprocesamiento de datos: {e}")
            return
        y_train = y_train.map({'SI': 0, 'NO': 1})
        y_test = y_test.map({'SI': 0, 'NO': 1})
           
        resultados = {}
        
        resultados['Arbol'] = optimizar_arbol_decision(X_train, y_train, X_test, y_test)

        modelos_guardados = {}
        
        if resultados:
            print("\nRESULTADOS DE OPTIMIZACIÓN:")
            print("-" * 70)
            print(f"{'Modelo':<20} | {'Mejores Parámetros':<30} | {'F1-Score (NO)':<12} | {'Exactitud':<10}")
            print("-" * 70)
            for modelo, datos in resultados.items():
                print(f"{modelo:<20}|{str(datos['mejores_params']):<30}|{datos['f1_no']:.4f}|{datos['exactitud']:.4f}")
                
                # Guardar el modelo
                ruta_modelo = f"src/models/{archivo}.joblib"
                joblib.dump({
                    'modelo': datos['mejor_modelo'],
                    'feature_names': X_train.columns.tolist()  
                    }, ruta_modelo)
                modelos_guardados[modelo] = ruta_modelo
                print(f"✅ Modelo guardado en: {ruta_modelo}")
            
            # Validación técnica para los 3 modelos guardados
            print("\nINICIANDO VALIDACIÓN TÉCNICA PARA LOS 3 MODELOS...")
            reportes_validacion = validacion_automatica(modelos_guardados, X_test, y_test)
            
            # Guardar metadatos de la ejecución
            guardar_metadatos(input_path, modelos_guardados, reportes_validacion)
            
            return resultados, modelos_guardados, reportes_validacion       
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return None

def guardar_metadatos(dataset_path, modelos_guardados, reportes):
    """Guarda metadatos de la ejecución"""
    ruta_absoluta = os.path.abspath(f"reports/metadata.txt")
    print(f"DEBUG: Intentando guardar en -> {ruta_absoluta}")
    os.makedirs(f"data/reports/", exist_ok=True)   
    try:
        with open(f"data/reports/metadata.txt", "w") as f:
            f.write(f"Fecha ejecución: {datetime.now()}\n")
            f.write(f"Dataset: {dataset_path}\n\n")
            
            f.write("Modelos optimizados:\n")
            for modelo, ruta in modelos_guardados.items():
                f.write(f"- {modelo}: {ruta}\n")
            
            f.write("\nResultados de validación:\n")
            for modelo, datos in reportes.items():
                f.write(f"\n{modelo}:\n")
                f.write(f"  Accuracy: {datos['predictivo']['accuracy']:.4f}\n")
                f.write(f"  F1-Score (NO): {datos['predictivo']['f1_NO']:.4f}\n")
                f.write(f"  AUC: {datos['predictivo'].get('auc', 'N/A')}\n")
    except Exception as e:
        print(f"ERROR al guardar metadatos: {e}")

def validacion_automatica(modelos_guardados, X_test, y_test):
    """Realiza validación técnica automática para todos los modelos guardados"""
    reportes = {}
    
    # Crear directorio para reportes
    report_dir = f"reports/"
    os.makedirs(report_dir, exist_ok=True)
    
    for modelo_nombre, ruta_modelo in modelos_guardados.items():
        try:
            print(f"\nVALIDANDO MODELO: {modelo_nombre}")
            obj = joblib.load(ruta_modelo)
            modelo  = obj['modelo']
            
            # Evaluación predictiva
            reporte = evaluar_modelo(modelo, X_test, y_test)

            # Guardar resultados
            reportes[modelo_nombre] = {
                'predictivo': reporte
            }
                        
        except Exception as e:
            print(f"❌ Error validando {modelo_nombre}: {str(e)}")
    
    return reportes

def evaluar_modelo(modelo, X_test, y_test):
    """Evaluación predictiva completa"""
    # Predicciones
    y_pred = modelo.predict(X_test)
    try:
        y_proba = modelo.predict_proba(X_test)[:, 1]
    except:
        y_proba = None
    
    # Métricas clave
    reporte = classification_report(y_test, y_pred, target_names=['SI', 'NO'], output_dict=True)
    
    resultados = {
        'accuracy': reporte['accuracy'],
        'f1_NO': reporte['NO']['f1-score'],
        'precision_NO': reporte['NO']['precision'],
        'recall_NO': reporte['NO']['recall'],
    }
    
    if y_proba is not None:
        resultados['auc'] = roc_auc_score(y_test, y_proba)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    resultados['matriz_confusion'] = cm.tolist()  # Convertir a lista para serialización
    
    return resultados

def optimizar_arbol_decision(X_train, y_train, X_test, y_test):
    """Optimiza hiperparámetros para Árbol de Decisión"""
    param_grid = {
        'max_depth': [7]
        , 'min_samples_split': [200]
        , 'min_samples_leaf': [100]
        , 'criterion': ['entropy']
        , 'ccp_alpha': [0.0005]
    }
    
    grid = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid,
        cv=StratifiedKFold(n_splits=5),
        scoring=f1_no_scorer,
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    mejor_modelo = grid.best_estimator_
    
    # Evaluar en conjunto de prueba
    y_pred = mejor_modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'mejores_params': grid.best_params_,
        'mejor_modelo': mejor_modelo,
        'f1_no': reporte['1']['f1-score'],
        'exactitud': reporte['accuracy']
    }

if __name__ == "__main__":
    run("com_si_no", "urbano")
