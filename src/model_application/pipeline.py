from src.model_application import arboles

def entrenar_arboles():

    while True:
        print("MENÚ DE MODELOS:")
        print("1 - Entrenar arboles")
        print("0 - Salir sin guardar")
        opcion = input("\nEscoge una opción (1-4 o 0): ")
        
        if opcion == '1':
            print("Entrenando arboles...")
            arboles.run()
            print("Optimización completada.")          
        elif opcion == '0':
            print("Saliendo")
            break
            
        else:
            print("Opción inválida. Intenta de nuevo.")