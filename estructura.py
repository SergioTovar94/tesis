import os

def imprimir_estructura(ruta_base, nivel=0, max_nivel=3):
    if nivel > max_nivel:
        return
    
    for nombre in sorted(os.listdir(ruta_base)):
        ruta = os.path.join(ruta_base, nombre)
        indentacion = "│   " * nivel + "├── " if nivel > 0 else ""
        print(f"{indentacion}{nombre}")
        
        if os.path.isdir(ruta):
            imprimir_estructura(ruta, nivel + 1, max_nivel)

if __name__ == "__main__":
    ruta_raiz = "."  # Cambia esto si estás en una subcarpeta
    imprimir_estructura(ruta_raiz)
