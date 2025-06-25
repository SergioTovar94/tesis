{'RL': 
    {'mejores_params': 
        {
        'C': 0.1, 
        'penalty': 'l1', 
        'solver': 'liblinear'
        }, 
    'mejor_modelo': 
        LogisticRegression(C=0.1, max_iter=1000, penalty='l1', solver='liblinear'), 
        'f1_no': 0.8720133667502089, 
        'exactitud': 0.9544481446241675
    }, 
'Arbol': 
    {'mejores_params': 
        {'ccp_alpha': 0.0005, 
        'criterion': 'gini', 
        'max_depth': 5, 
        'min_samples_leaf': 100, 
        'min_samples_split': 200
        }, 
    'mejor_modelo': 
        DecisionTreeClassifier(ccp_alpha=0.0005, max_depth=5, min_samples_leaf=100,min_samples_split=200), 
        'f1_no': 0.8839187005697764, 
        'exactitud': 0.9594136536631779
    }, 
'MLP': {'mejores_params': {
    'activation': 'tanh', 
    'alpha': 0.01, 
    'early_stopping': True, 
    'hidden_layer_sizes': (6,), 'learning_rate_init': 0.01, 'max_iter': 500}, 
    'mejor_modelo': MLPClassifier(activation='tanh', alpha=0.01, early_stopping=True,
            hidden_layer_sizes=(6,), learning_rate_init=0.01, max_iter=500), 
            'f1_no': 0.8715933790336065, 
            'exactitud': 0.9543292102759277
        }
}