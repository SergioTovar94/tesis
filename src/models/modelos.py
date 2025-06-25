import os
import joblib
from datetime import datetime
from sklearn.metrics import roc_auc_score,confusion_matrix
from src import config

def guardar_metadatos(dataset_path, modelos_guardados, reportes):
    """Guarda metadatos de la ejecución"""
    ruta_absoluta = os.path.abspath(f"{config.CARPETA_REPORTES}/{config.ZONA_POR_DEFECTO}/metadata.txt")
    print(f"DEBUG: Intentando guardar en -> {ruta_absoluta}")
    try:
        with open(f"{config.CARPETA_REPORTES}/{config.ZONA_POR_DEFECTO}/metadata.txt", "w") as f:
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

    report_dir = os.path.join(config.CARPETA_REPORTES, config.ZONA_POR_DEFECTO)
    os.makedirs(report_dir, exist_ok=True)
    
    for modelo_nombre, ruta_modelo in modelos_guardados.items():
        try:
            print(f"\nVALIDANDO MODELO: {modelo_nombre}")
            obj = joblib.load(ruta_modelo)
            modelo  = obj['modelo']

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



def run(archivo: str):

    # Validación técnica para los 3 modelos guardados
    print("\nINICIANDO VALIDACIÓN TÉCNICA PARA LOS 3 MODELOS...")
    reportes_validacion = validacion_automatica(modelos_guardados, X_test, y_test)
            
    # Guardar metadatos de la ejecución
    guardar_metadatos(input_path, modelos_guardados, reportes_validacion)
            
    return resultados, modelos_guardados, reportes_validacion       


