import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.io_utils import cargar_dataset
from utils.modelo_utils import preprocess_data, seleccionar_columnas
import shap

def validacion_post_optimizacion(carpeta: str, zona: str):
    try:
        # 1. Cargar datos
        input_path = f"data/processed/{carpeta}/Dataset_{zona}_col_cal_smote.csv"
        df = cargar_dataset(input_path)
        
        # 2. Preprocesamiento
        X_train, X_test, y_train, y_test = preprocess_data(df)
        X_train, X_test = seleccionar_columnas(X_train, X_test)
        
        # Convertir etiquetas
        y_test = y_test.map({'SI': 0, 'NO': 1})
        
        # 3. Cargar modelo
        ruta_modelo = f"src/models/{modelo}_{zona}.joblib"
        modelo = joblib.load(ruta_modelo)
        print(f"✅ Modelo cargado desde: {ruta_modelo}")
        
        # 4. Evaluación predictiva
        print("\nEVALUACIÓN PREDICTIVA:")
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None
        
        print(classification_report(y_test, y_pred, target_names=['SI', 'NO']))
        
        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            print(f"AUC-ROC: {auc:.4f}")
        
        # 5. Análisis de interpretabilidad
        print("\nANÁLISIS DE INTERPRETABILIDAD:")
        if isinstance(modelo, LogisticRegression):
            # Coeficientes para Regresión Logística
            coefs = pd.DataFrame({
                'Variable': X_test.columns,
                'Coeficiente': modelo.coef_[0],
                'Odds_Ratio': np.exp(modelo.coef_[0])
            }).sort_values('Odds_Ratio', ascending=False)
            
            print("\nCoeficientes y Odds Ratios:")
            print(coefs)
            
            # Gráfico
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Odds_Ratio', y='Variable', data=coefs.head(10))
            plt.title('Top 10 Variables - Impacto en Probabilidad de NO Pago')
            plt.savefig(f"reports/figures/{zona}_odds_ratios.png")
            plt.show()
            
        elif isinstance(modelo, (DecisionTreeClassifier)):
            # Importancia de características para árboles
            if hasattr(modelo, 'feature_importances_'):
                importancia = pd.DataFrame({
                    'Variable': X_test.columns,
                    'Importancia': modelo.feature_importances_
                }).sort_values('Importancia', ascending=False)
                
                print("\nImportancia de Variables:")
                print(importancia.head(10))
                
                # Gráfico
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Importancia', y='Variable', data=importancia.head(10))
                plt.title('Top 10 Variables - Importancia')
                plt.savefig(f"reports/figures/{zona}_feature_importance.png")
                plt.show()
            
            # Para árboles de decisión: extraer reglas
            if isinstance(modelo, DecisionTreeClassifier):
                from sklearn.tree import export_text
                tree_rules = export_text(modelo, feature_names=list(X_test.columns))
                with open(f"reports/tree_rules_{zona}.txt", "w") as f:
                    f.write(tree_rules)
                print(f"✅ Reglas del árbol guardadas en: reports/tree_rules_{zona}.txt")
        
        elif isinstance(modelo, MLPClassifier):
            # SHAP para redes neuronales
            explainer = shap.KernelExplainer(modelo.predict_proba, shap.sample(X_train, 100))
            shap_values = explainer.shap_values(X_test.iloc[:10])
            
            plt.figure()
            shap.summary_plot(shap_values[1], X_test.iloc[:10], plot_type="bar")
            plt.title('Importancia de Variables (MLP)')
            plt.savefig(f"reports/figures/{zona}_shap_summary.png")
            plt.show()
        
        # 6. Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['SI', 'NO'], 
                    yticklabels=['SI', 'NO'])
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.title('Matriz de Confusión')
        plt.savefig(f"reports/figures/{zona}_confusion_matrix.png")
        plt.show()
        
        print("\n✅ Validación técnica completada con éxito!")
        
    except Exception as e:
        print(f"\n❌ ERROR en validación post-optimización: {str(e)}")

if __name__ == "__main__":
    # Ejemplo de uso:
    validacion_post_optimizacion(
        carpeta="com_si_no",
        zona="urbano",
        ruta_modelo="src/models/RL_urbano_20230615_143045.joblib"
    )