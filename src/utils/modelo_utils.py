import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def preprocess_data(df):
    X = df.drop(columns=['COMPORTAMIENTO_PAGO'])  
    y = df['COMPORTAMIENTO_PAGO']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No Pagó", "Sí Pagó"])
    
    print(f"\n📊 Exactitud del modelo: {accuracy:.4f}")
    print("\n🔍 Informe de Clasificación:\n")
    print(report)

def seleccionar_columnas(X_train, X_test):
    col_a_eliminar = X_train.columns[2]
    for i in range(5):
        col_a_eliminar = X_train.columns[4]
        X_train = X_train.drop(columns=[col_a_eliminar])
        X_test = X_test.drop(columns=[col_a_eliminar])
    while len(X_train.columns) > 1:
        print("\n🔍 Columnas disponibles para el entrenamiento:")
        for i, col in enumerate(X_train.columns, 1):
            print(f"{i}. {col}")
        print("\nSi deseas eliminar una columna, ingresa su número.")
        print("Si deseas continuar con estas columnas, ingresa 0.")

        try:
            opcion = int(input("\nSelecciona una opción: ").strip())
            if opcion == 0:
                break
            elif 1 <= opcion <= len(X_train.columns):
                col_a_eliminar = X_train.columns[opcion - 1]
                X_train = X_train.drop(columns=[col_a_eliminar])
                X_test = X_test.drop(columns=[col_a_eliminar])
                print(f"✅ Columna '{col_a_eliminar}' eliminada.")
            else:
                print("❌ Opción no válida. Intenta de nuevo.")
        except ValueError:
            print("❌ Ingresa un número válido.")

    return X_train, X_test
