from utils.modelo_utils import preprocess_data, evaluate_model, seleccionar_columnas
from src.data.oi_utils import cargar_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os

def seleccionar_modelo():
    modelos = {
        "1": ("Regresión Logística", LogisticRegression()),
        "2": ("Árbol de Decisión", DecisionTreeClassifier()),
        "3": ("Random Forest", RandomForestClassifier(n_estimators=100)),
        "4": ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    }
    
    while True:
        print("\nSelecciona el modelo a entrenar:")
        for key, (name, _) in modelos.items():
            print(f"{key}⃣ {name}")
        print("0️⃣ Salir")
        
        opcion = input("\n👉 Opción: ").strip()
        if opcion in modelos:
            return modelos[opcion]
        elif opcion == "0":
            return None, None
        else:
            print("❌ Opción no válida. Intenta de nuevo.")

def run(carpeta: str, zona: str):
    input_path = f"data/processed/{carpeta}/4_Dataset_{zona}_col_cal_sin_outliers_estandarizado.csv"
    
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
    
    while True:
        modelo_nombre, modelo = seleccionar_modelo()
        if modelo is None:
            print("👋 Finalizando el programa.")
            break
        
        print(f"\n📌 Modelo seleccionado: {modelo_nombre}")

        try:
            modelo.fit(modelo, X_train, y_train)
            accuracy, report = evaluate_model(modelo, X_test, y_test)
        except Exception as e:
            print(f"❌ Error durante el entrenamiento o evaluación: {e}")
            continue
        
        print(f"\n📊 Exactitud del modelo: {accuracy:.4f}")
        print("\n🔍 Informe de Clasificación:\n", report)
        
        if input("\n🔄 ¿Quieres entrenar otro modelo? (1 = Sí, 0 = No): ").strip() != "1":
            print("👋 Finalizando el programa.")
            break


if __name__ == "__main__":
    run("com_si_no", "urbano")
