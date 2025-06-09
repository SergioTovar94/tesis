from src.features import (
    standardize,
)
from src.features import delete_outliers
from src.features import feature_engineering, generate_target
from src.models import apply_smote, split_dataset
from src.preprocessing import initial_filter, panel_to_cross_section
from src.preprocessing import clean_data
from src.data.io_utils import actualizar_para_weka

def alistar_datasets(carpeta: str):
    
    initial_filter.run(carpeta)

    panel_to_cross_section.run(carpeta)

    clean_data.run(carpeta)

    generate_target.run(2019, "urbano", carpeta)

    actualizar_para_weka()