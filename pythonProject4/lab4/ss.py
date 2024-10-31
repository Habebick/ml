from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('german.csv', sep=';')

X = data.iloc[:, 1:]
y = data.iloc[:, 0].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'hidden_layer_sizes': [(30,), (50, 50), (100,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [0.0001, 0.001, 0.01]
}

grid_search = GridSearchCV(estimator=MLPClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='roc_auc',
                           cv=5)

grid_search.fit(X_train, y_train)
gb_model = GradientBoostingClassifier(random_state=42)
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}
grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=param_grid_gb, scoring='roc_auc', cv=5)
grid_search_gb.fit(X_train, y_train)
best_gb_model = GradientBoostingClassifier(**grid_search_gb.best_params_, random_state=42)
best_gb_model.fit(X_train, y_train)
gb_pred = best_gb_model.predict(X_test)
gb_roc_auc = roc_auc_score(y_test, gb_pred)
print(f"Gradient Boosting (Optimized) ROC AUC: {gb_roc_auc:.2f}")
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

