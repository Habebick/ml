from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, accuracy_score

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_csv('german.csv', sep=';')


X = data.iloc[:, 1:].to_numpy()
y = data.iloc[:, 0].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
plt.hist(y_train, bins=2, edgecolor='k')
plt.xticks([0, 1])
plt.xlabel('Class (0: Non-Creditworthy, 1: Creditworthy)')
plt.ylabel('Count')
plt.title('Distribution of Classes in Training Data')
#plt.show()
# Обучение Random Forest
rf_model = RandomForestClassifier(n_estimators=50, max_depth=45, min_samples_split=15, min_samples_leaf=2, random_state=42)
rf_model.fit(X_train, y_train)

# Прогноз на тестовых данных
rf_pred = rf_model.predict(X_test)

# Расчет метрик для Random Forest
rf_roc_auc = roc_auc_score(y_test, rf_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)

print("Random Forest метрики:")
print(f"ROC AUC: {rf_roc_auc:.2f}")
#print(f"Accuracy: {rf_accuracy:.2f}")
#print(f"Precision: {rf_precision:.2f}")
#print(f"Recall: {rf_recall:.2f}")
#Best Hyperparameters: {'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (100,)}

# Обучение Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Прогноз на тестовых данных
gb_pred = gb_model.predict(X_test)

# Расчет метрик для Gradient Boosting
gb_roc_auc = roc_auc_score(y_test, gb_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)
gb_precision = precision_score(y_test, gb_pred)
gb_recall = recall_score(y_test, gb_pred)

print("\nGradient Boosting метрики:")
print(f"ROC AUC: {gb_roc_auc:.2f}")
#print(f"Accuracy: {gb_accuracy:.2f}")
#print(f"Precision: {gb_precision:.2f}")
#print(f"Recall: {gb_recall:.2f}")
# Обучение MLP (Multi-Layer Perceptron) нейронной сети
mlp_model = MLPClassifier(hidden_layer_sizes=(30,), alpha=0.01, activation='logistic', max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

# Прогноз на тестовых данных
mlp_pred = mlp_model.predict(X_test)

# Расчет метрик для MLP нейронной сети
mlp_roc_auc = roc_auc_score(y_test, mlp_pred)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
mlp_precision = precision_score(y_test, mlp_pred)
mlp_recall = recall_score(y_test, mlp_pred)

print("\nMLP (Neural Network) метрики:")
print(f"ROC AUC: {mlp_roc_auc:.2f}")
#print(f"Accuracy: {mlp_accuracy:.2f}")ффффффффффыыыыыыыыыыыыыыыфыуц
#print(f"Precision: {mlp_precision:.2f}")
#print(f"Recall: {mlp_recall:.2f}")
for model in [rf_model, gb_model, mlp_model]:
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"{model.__class__.__name__} ROC AUC: {scores.mean():.2f}")


ensemble_model = StackingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model),
        ('mlp', mlp_model)
    ],
    final_estimator=GradientBoostingClassifier(n_estimators=200, random_state=42)
)

ensemble_model.fit(X_train, y_train)
ensemble_pred = ensemble_model.predict(X_test)

# Расчет метрик для ансамбля
ensemble_roc_auc = roc_auc_score(y_test, ensemble_pred)

print("\nАнсамбль (Stacking) метрики:")
print(f"ROC AUC: {ensemble_roc_auc:.2f}")
