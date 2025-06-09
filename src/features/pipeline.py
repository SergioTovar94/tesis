from standardize import escalar_datos
from src.features import delete_outliers

def transformar_dataset(carpeta: str, archivo: str):
    
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
            df = escalar_datos(carpeta, archivo, metodo='zscore')
            print("Escalado aplicado.")
            
        elif opcion == '2':
            print("Aplicando RobustScaler...")
            df = escalar_datos(carpeta, archivo, metodo='robust')
            print("Escalado aplicado.")
            
        elif opcion == '3':
            print("Eliminando outliers con método IQR...")
            df = delete_outliers(carpeta, archivo)
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
