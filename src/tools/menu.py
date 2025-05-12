import os

def seleccionar_opcion(menu, titulo):
    """Muestra un menú y retorna la opción seleccionada."""
    while True:
        print(f"\n📌 {titulo}:")
        for key, value in menu.items():
            print(f"{key}️⃣ {value}")
        print("0️⃣ Cancelar")
        
        opcion = input("\n👉 Selecciona una opción: ").strip()
        if opcion == "0":
            return None
        if opcion in menu:
            return menu[opcion]
        print("❌ Opción no válida. Intenta de nuevo.")

def seleccionar_archivo(carpeta):
    """
    Lista los archivos en una carpeta y permite al usuario seleccionar uno.

    Args:
        carpeta (str): Ruta de la carpeta.

    Returns:
        str: Nombre del archivo seleccionado o None si no hay archivos.
    """
    try:
        carpeta = f'data/processed/{carpeta}'
        archivos = os.listdir(carpeta)
        archivos = [archivo for archivo in archivos if os.path.isfile(os.path.join(carpeta, archivo))]

        if not archivos:
            print(f"❌ No se encontraron archivos en la carpeta: {carpeta}")
            return None

        print("\n📂 Archivos disponibles:")
        for i, archivo in enumerate(archivos, 1):
            print(f"{i}. {archivo}")

        opcion = input("\n👉 Selecciona un archivo por número: ").strip()
        if opcion.isdigit() and 1 <= int(opcion) <= len(archivos):
            return archivos[int(opcion) - 1]
        else:
            print("❌ Opción no válida.")
            return None

    except FileNotFoundError:
        print(f"❌ La carpeta {carpeta} no existe.")
        return None