from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score
import config

f1_no_scorer = make_scorer(f1_score, pos_label=1)

def optimizar_regresion_logistica(X_train, y_train, X_test, y_test):
    """Optimiza hiperparámetros para Regresión Logística"""   
    
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        config.GRID_REGRESION_LOGISTICA,
        cv=StratifiedKFold(n_splits=5),
        scoring=f1_no_scorer,  # Priorizamos F1-Score
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    mejor_modelo = grid.best_estimator_
    
    # Evaluar en conjunto de prueba
    y_pred = mejor_modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'mejores_params': grid.best_params_,
        'mejor_modelo': mejor_modelo,
        'f1_no': reporte['1']['f1-score'],
        'exactitud': reporte['accuracy']
    }

def optimizar_arbol_decision(X_train, y_train, X_test, y_test):
    """Optimiza hiperparámetros para Árbol de Decisión"""

    
    grid = GridSearchCV(
        DecisionTreeClassifier(),
        config.GRID_ARBOL,
        cv=StratifiedKFold(n_splits=5),
        scoring=f1_no_scorer,
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    mejor_modelo = grid.best_estimator_
    
    # Evaluar en conjunto de prueba
    y_pred = mejor_modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'mejores_params': grid.best_params_,
        'mejor_modelo': mejor_modelo,
        'f1_no': reporte['1']['f1-score'],
        'exactitud': reporte['accuracy']
    }

def optimizar_mlp(X_train, y_train, X_test, y_test):
    """Optimiza hiperparámetros para Perceptrón Multicapa"""

    grid = GridSearchCV(
        MLPClassifier(max_iter=1000),
        config.GRID_MLP,
        cv=StratifiedKFold(n_splits=3),  # Menos folds por costo computacional
        scoring=f1_no_scorer,
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    mejor_modelo = grid.best_estimator_
    
    # Evaluar en conjunto de prueba
    y_pred = mejor_modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'mejores_params': grid.best_params_,
        'mejor_modelo': mejor_modelo,
        'f1_no': reporte['1']['f1-score'],
        'exactitud': reporte['accuracy']
    }
