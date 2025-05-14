import pandas as pd

def eliminar_nan_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permite ver las columnas con valores NaN y eliminar los NaN de la columna seleccionada sobre un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a analizar y limpiar.

    Returns:
        pd.DataFrame: DataFrame limpio según la columna seleccionada.
    """
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if nan_cols.empty:
        print("✅ No hay columnas con valores NaN en el DataFrame.")
        return df

    print("\n🔍 Columnas con valores NaN:")
    for i, (col, count) in enumerate(nan_cols.items(), 1):
        print(f"{i}. {col}: {count} NaN")

    try:
        opcion = int(input("\n👉 Ingresa el número de la columna a la que deseas eliminar los NaN (0 para cancelar): ").strip())
        if opcion == 0:
            print("🔙 Operación cancelada.")
            return df
        elif 1 <= opcion <= len(nan_cols):
            col_a_limpiar = nan_cols.index[opcion - 1]
            df = df.dropna(subset=[col_a_limpiar])
            print(f"✅ Se eliminaron los registros con NaN en la columna '{col_a_limpiar}'.")
            return df
        else:
            print("❌ Opción no válida.")
            return df
    except ValueError:
        print("❌ Ingresa un número válido.")
        return df