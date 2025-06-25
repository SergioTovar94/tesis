from src.preprocessing import initial_filter, panel_to_cross_section, clean_data
from src.features import generate_target
from src.utils.io_utils import ubicar_en_raiz

def alistar_datasets():
    
    initial_filter.run()

    panel_to_cross_section.run()

    clean_data.run()

    generate_target.run()

    ubicar_en_raiz()