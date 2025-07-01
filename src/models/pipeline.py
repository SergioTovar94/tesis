from src.models import optimizar_modelos, imprimir_modelos, calcular_interpretabilidad

def entrenar_modelos(archivo: str):

    while True:
        print("MENÚ DE MODELOS:")
        print("1 - Optimizar modelos")
        print("2 - Imprimir modelo")
        print("3 - Calcular interpretabilidad")
        print("0 - Salir sin guardar")
        opcion = input("\nEscoge una opción (1-4 o 0): ")
        
        if opcion == '1':
            print("Aplicando optimización de modelos...")
            optimizar_modelos.run(archivo)
            print("Optimización completada.")
            
        elif opcion == '2':
            print("Imprimiendo modelo...")
            imprimir_modelos.run()
            print("Impresión completada.")
        elif opcion == '3':
            print("Calculando interpretabilidad...")
            calcular_interpretabilidad.run()
            print("Interpretabilidad calculada.")            
        elif opcion == '0':
            print("Saliendo")
            break
            
        else:
            print("Opción inválida. Intenta de nuevo.")

    


    
    