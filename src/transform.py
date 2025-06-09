import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from src.data.io_utils import cargar_dataset, guardar_dataset

def escalar_datos(df, columnas_numericas, metodo='zscore'):
    if metodo == 'zscore':
        scaler = StandardScaler()
    elif metodo == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Método debe ser 'zscore' o 'robust'")

    ct = ColumnTransformer(
        transformers=[('scaler', scaler, columnas_numericas)],
        remainder='passthrough'
    )
    datos_escalados = ct.fit_transform(df)
    columnas_passthrough = [col for col in df.columns if col not in columnas_numericas]
    columnas_ordenadas = columnas_numericas + columnas_passthrough
    df_escalado = pd.DataFrame(datos_escalados, columns=columnas_ordenadas)
    df_escalado = df_escalado[df.columns]
    return df_escalado

def aplicar_smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def transformar_dataset(dataset: str, carpeta: str):
    input_path = f"data/processed/{carpeta}/Dataset_{dataset}.csv"
    output_path = f"data/processed/{carpeta}/Dataset_{dataset}_estandarizado.csv"

    df = cargar_dataset(input_path)

    print("MENÚ DE TRANSFORMACIONES:")
    print("1 - Escalar con Z-score (StandardScaler)")
    print("2 - Escalar con RobustScaler")
    print("3 - Eliminar outliers (IQR)")
    print("4 - Aplicar SMOTE (solo a features + target)")
    print("5 - Guardar dataset y salir")
    print("0 - Salir sin guardar")
    
    while True:
        opcion = input("\nEscoge una opción (1-5 o 0): ")
        
        if opcion == '1':
            print("Aplicando Z-score...")
            df = escalar_datos(df, columnas_num, metodo='zscore')
            print("Escalado aplicado.")
            
        elif opcion == '2':
            print("Aplicando RobustScaler...")
            df = escalar_datos(df, columnas_num, metodo='robust')
            print("Escalado aplicado.")
            
        elif opcion == '3':
            print("Eliminando outliers con método IQR...")
            df = eliminar_outliers_iqr(df, columnas_num)
            print(f"Outliers eliminados. Nuevo tamaño del dataset: {len(df)}")
            
        elif opcion == '4':
            print("Aplicando SMOTE para balancear clases...")
            X = df.drop(columns=[columna_objetivo])
            y = df[columna_objetivo]
            X_res, y_res = aplicar_smote(X, y)
            df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=columna_objetivo)], axis=1)
            print(f"SMOTE aplicado. Nuevo tamaño del dataset: {len(df)}")
            
        elif opcion == '5':
            nombre_archivo = input("Escribe el nombre para guardar el CSV (ejemplo: dataset_procesado.csv): ")
            df.to_csv(nombre_archivo, index=False)
            print(f"Dataset guardado en {nombre_archivo}. Saliendo...")
            break
            
        elif opcion == '0':
            print("Saliendo sin guardar.")
            break
            
        else:
            print("Opción inválida. Intenta de nuevo.")
        
        print("\nVista previa del dataset:")
        print(df.head())

if __name__ == "__main__":
    main()
