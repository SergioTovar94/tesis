from preprocessings import (
    step1_generate_bd_initial,
    step2_transform_panel_to_transversal,
    step3_comportamiento_pago,
    step4_columnas_calculadas,
    step5_sin_outliers,
    step6_equilibrado,
    step7_estandarizado,
)
from src.utils.print_utils import print_message
from src.tools import cruzar_dataset
from src.utils.oi_utils import actualizar_para_weka

def alistar_datasets(carpeta: str):
    """Función para alistar los datasets."""
    print_message(1, "Generando base inicial")
    step1_generate_bd_initial.run(carpeta)

    print_message(2, "Transformando panel a transversal")
    step2_transform_panel_to_transversal.run(carpeta)

    print_message(3, "Generando columna de comportamiento de pago")
    step3_comportamiento_pago.run(2019, "urbano", carpeta)
    step3_comportamiento_pago.run(2020, "rural", carpeta)

    print_message(3.1, "Eliminando outliers")
    step5_sin_outliers.run("3_Dataset_urbano", carpeta)

    print_message(7, "Estandarizando datos")
    step7_estandarizado.run("3_Dataset_urbano", carpeta)

    print_message(7, "Estandarizando datos")
    step7_estandarizado.run("3_Dataset_urbano_sin_outliers", carpeta)

    #print_message(99, "Cruzando dataset")
    #cruzar_dataset.run(carpeta)

    print_message(4, "Generando columnas calculadas")
    step4_columnas_calculadas.run(carpeta, "urbano")

    print_message(5, "Eliminando outliers")
    step5_sin_outliers.run("4_Dataset_urbano_col_cal_2", carpeta)

    print_message(6, "Equilibrando datos")
    step6_equilibrado.run("4_Dataset_urbano_col_cal_2_sin_outliers", carpeta)

    print_message(7, "Estandarizando datos")
    step7_estandarizado.run("4_Dataset_urbano_col_cal_2_sin_outliers", carpeta)

    print_message(8, "Equilibrando después de estandarizar")
    step6_equilibrado.run("4_Dataset_urbano_col_cal_2_sin_outliers_estandarizado", carpeta)

    actualizar_para_weka()