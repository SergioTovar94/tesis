from src.preprocessing import (
    standardize,
)
from src.data import delete_outliers
from src.features import feature_engineering, generate_target
from src.models import apply_smote
from src.preprocessing import initial_filter, panel_to_cross_section, split_dataset
from src.preprocessing import clean_data
from src.data.io_utils import actualizar_para_weka

def alistar_datasets(carpeta: str):
    
    """initial_filter.run(carpeta)

    panel_to_cross_section.run(carpeta)

    clean_data.run(carpeta)

    generate_target.run(2019, "urbano", carpeta)

    delete_outliers.run("urbano", carpeta)
    
    feature_engineering.run("urbano", carpeta)

    feature_engineering.run("urbano_sin_outliers", carpeta)

    apply_smote.run("urbano", carpeta)

    apply_smote.run("urbano_col_cal", carpeta)

    standardize.run("urbano_col_cal", carpeta)

    apply_smote.run("urbano_sin_outliers_col_cal", carpeta)

    split_dataset.run("urbano_col_cal", carpeta)

    split_dataset.run("urbano_sin_outliers_col_cal", carpeta)

    split_dataset.run("urbano_sin_outliers_col_cal_smote", carpeta)"""

    split_dataset.run("urbano_col_cal_smote", carpeta)
    

    actualizar_para_weka()