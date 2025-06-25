from src.models import optimizar_modelos, analizar_modelos

def entrenar_modelos(archivo: str):

    while True:
        print("MENÚ DE MODELOS:")
        print("1 - Optimizar modelos")
        print("2 - Evaluar modelos optimizados")
        print("0 - Salir sin guardar")
        opcion = input("\nEscoge una opción (1-4 o 0): ")
        
        if opcion == '1':
            print("Aplicando optimización de modelos...")
            optimizar_modelos.run(archivo)
            print("Optimización completada.")
            
        elif opcion == '2':
            print("Evaluando modelos optimizados...")
            analizar_modelos.run(archivo)
            print("Evaluación completada.")            
        elif opcion == '0':
            print("Saliendo")
            break
            
        else:
            print("Opción inválida. Intenta de nuevo.")

    


    
    