import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src import config

def preprocess_data(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=[config.VARIABLE_OBJETIVO])  
    y = df[config.VARIABLE_OBJETIVO]  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.dtypes)
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    y_train = y_train.map({'SI': 0, 'NO': 1})
    y_test = y_test.map({'SI': 0, 'NO': 1})
    return X_train, X_test, y_train, y_test

def seleccionar_columnas(X_train, X_test):
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


