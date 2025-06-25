from src.features import standardize, delete_outliers, smote

def transformar_dataset(archivo: str):
    
    while True:
        print("MENÚ DE TRANSFORMACIONES:")
        print("1 - Escalar con Z-score (StandardScaler)")
        print("2 - Escalar con RobustScaler")
        print("3 - Eliminar outliers (IQR)")
        print("4 - Aplicar SMOTE (solo a features + target)")
        print("0 - Salir sin guardar")
        opcion = input("\nEscoge una opción (1-4 o 0): ")
        
        if opcion == '1':
            print("Aplicando Z-score...")
            standardize.run(archivo, metodo='zscore')
            print("Escalado aplicado.")
            
        elif opcion == '2':
            print("Aplicando RobustScaler...")
            standardize.run(archivo, metodo='robust')
            print("Escalado aplicado.")
            
        elif opcion == '3':
            print("Eliminando outliers con método IQR...")
            delete_outliers.run(archivo)
            
        elif opcion == '4':
            print("Aplicando SMOTE para balancear clases...")
            smote.run(archivo)            
        elif opcion == '0':
            print("Saliendo")
            break
            
        else:
            print("Opción inválida. Intenta de nuevo.")

