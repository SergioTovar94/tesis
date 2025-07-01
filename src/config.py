# src/config.py

# -------------------------------
# VARIABLES GLOBALES DEL PROYECTO
# -------------------------------

CARPETAS_DATASETS = {
    "1": "com_dejo_pagar",
    "2": "com_si_no"
}

ZONAS_DISPONIBLES = {
    "1": "urbano",
    "2": "rural",
}

ESTADISTICOS = {
    "1": "Calcular matriz de correlación",
    "2": "Calcular VIF (Factor de Inflación de Varianza)",
}

MENU_MODELOS = {
    "1": "Regresión Logística",
    "2": "Árbol de Decisión",
    "3": "Perceptrón Multicapa (MLP)",
    "4": "Optimizar los 3 modelos"
}

# -------------------------------
# CONFIGURACIÓN ESCOGIDA
# -------------------------------
LLAVE = 'NUMPRED'
TIPO_VARIABLE_OBJETIVO = 'com_si_no'
ZONA_POR_DEFECTO = 'urbano'
ANIO_URBANO = 2022
ANIO_RURAL = 2019

# -------------------------------
# VARIABLES
# -------------------------------

VARIABLE_OBJETIVO = 'COMPORTAMIENTO_PAGO'
COLUMNAS_FIJAS = ['NUMPRED', 'TIPO_PREDIO', 'ESTRATO', 'AREA_CONSTRUIDA', 'BARRIO']
COLUMNAS_A_PIVOTEAR = ['AVALUO_CATASTRAL','IMPUESTO_PREDIAL_BRUTO', 'IMPUESTO_PREDIAL_APLICADO', 'RECIPU']
PIVOTE = ['VIGENCIA']
COLUMNAS_EXCLUIR_OUTLIERS = ['RECIPU', 'ANIOS']
IQR_FACTOR = 1.5

# -------------------------------
# RUTAS PRINCIPALES CARPETAS
# -------------------------------

DATA_PROCESSED = 'data/processed/'
DATA_GRAPHS = 'data/graphs/'
DATA_RAW = 'data/raw/'
CARPETA_LOCAL = 'C:\\Users\\sergi\\OneDrive\\Documentos'
CARPETA_MODELOS = 'models/'
CARPETA_REPORTES = 'reports/'
MATRICES = 'matrices/'
ARBOL = 'arbol/'

# -------------------------------
# RUTAS PRINCIPALES ARCHIVOS
# -------------------------------

DATA_ORIGINAL = 'LIQ_IPU_CONS_2024.csv'
PANEL_DEPURADA = "Dataset_panel_depurada.csv"
TRANSVERSAL = 'Dataset_transversal.csv'
TRANSVERSAL_DEPURADA = 'Dataset_transversal_depurada.csv'
URBANO = 'Dataset_urbano.csv'
# -------------------------------
# CONFIGURACIÓN DE LIMPIEZA DE DATOS
# -------------------------------

ANIO_OBJETIVO = 2019
ZONA_POR_DEFECTO = 'urbano'
ESTRATOS_ATIPICOS = [0, 8, 9]

# -------------------------------
# NORMATIVA
# -------------------------------

ACTUALIZACION_URBANA = 2020
ACTUALIZACION_RURAL = 2021

SALARIO_MINIMO_2019 = 828116
SALARIO_MINIMO_2020 = 877803
SALARIO_MINIMO_2021 = 908526

# Tarifas para estatuto viejo
TARIFAS_ANTIGUAS = [
    (120, 0.0055),
    (200, 0.0065),
    (300, 0.007),
    (float('inf'), 0.0075)
]

# Tarifas para estatuto nuevo
TARIFAS_NUEVAS = [
    (27, 0.005),
    (62, 0.006),
    (135, 0.007),
    (180, 0.008),
    (269, 0.009),
    (414, 0.010),
    (float('inf'), 0.016)
]

# -------------------------------
# VARIABLES A ELIMINAR Y FILTROS
# -------------------------------

COLUMNAS_A_ELIMINAR = [
    'DESTINO_ECONOMICO', 'ESTRATO_SOCIAL', 'TARIFA_PREDIAL', 'FACTCAR','LIQIPU',
    'MORA', 'DEP_MORA', 'MORA_DEF', 'CANTMORA', 'MORATOT'
]

ANIOS_A_FILTRAR = [2022, 2023, 2024]

DESTINOS_A_FILTRAR = [
        "NULO", "AGROINDUSTRIAL", "SERVICIO FUNERARIO", "AGROFORESTAL", "CULTURAL", "SALUBRIDAD",
        "INFRAESTRUCTURA HIDRICA", "PECUARIO", "FORESTAL", "LOTE NO URBANIZABLE (ServEsp)", "RECREACIONAL",
        "INSTITUCIONAL", "MINERO", "RELIGIOSO", "AGRICOLA", "INDUSTRIAL", "EDUCATIVO", "USO PUBLICO"
    ]

GRID_REGRESION_LOGISTICA = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1],
    'solver': ['liblinear']
}

GRID_ARBOL = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [200],
    'min_samples_leaf': [100],
    'criterion': ['gini', 'entropy'],
    'ccp_alpha': [0.0005]
}

GRID_MLP = {
    'hidden_layer_sizes': [(6,), (6, 6), (6, 10, 6), (6, 50, 6), (10,10)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.01, 0.1],
    'learning_rate_init': [0.01],
    'early_stopping': [True],
    'max_iter': [500]
}

RUTA_ARBOLES = 'data/processed/com_si_no/arboles/'

COLUMNAS_DEF = [
    "ANIOS_PAGADOS",
    "AREA_CONSTRUIDA",
    "VARIACION_AVALUO",
    "VARIACION_TARIFA",
    "DESCUENTO",
    "ESTRATO"
]