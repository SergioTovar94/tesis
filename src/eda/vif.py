from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def calcular_vif(df):
    """
    Calcula el Factor de Inflación de Varianza (VIF) para un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con variables numéricas.

    Returns:
        pd.DataFrame: DataFrame con las variables y sus respectivos VIF.
    """
    x = df.select_dtypes(include=['number'])
    vif_data = pd.DataFrame()
    vif_data["Variable"] = x.columns
    vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    return vif_data