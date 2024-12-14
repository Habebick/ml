import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


# Загружаем данные и заполняем пустые строчки (0)
data = pd.read_csv('dataset.csv')
data = data.fillna(0)

# Создаем новые колонки
data['duration_min'] = data['duration_ms'] / 60000
data["loudness_scaled"] = data["loudness"] / data["loudness"].max()
data['tempo_scaled'] = data["tempo"] / data["tempo"].max()

# Разделяем данные на признаки X и ключевую переменную y (без track_id)
X = data[['artists', "album_name", "track_name", "duration_ms", "explicit", "danceability", "energy", "key",
          "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo",
          "time_signature", "track_genre"]].copy()

# Добавляем новые колонки в X
X['duration_min'] = data['duration_min']
X['loudness_scaled'] = data['loudness_scaled']
X['tempo_scaled'] = data['tempo_scaled']

# Явное преобразование новых колонок в числовой тип
X['duration_min'] = pd.to_numeric(X['duration_min'], errors='coerce').fillna(0)
X['loudness_scaled'] = pd.to_numeric(X['loudness_scaled'], errors='coerce').fillna(0)
X['tempo_scaled'] = pd.to_numeric(X['tempo_scaled'], errors='coerce').fillna(0)

y = data['popularity']


# Информация о данных
print("Data shape:", data.shape)
print("\nData types:\n", data.dtypes)
print("\nMissing values per column:\n", data.isnull().sum())
print("\nDescriptive stats:\n", data.describe())

# Распределение целевой переменной
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Distribution of Target Variable ('popularity')")
plt.xlabel("Popularity (0: No, 1: Yes)")
plt.ylabel("Count")
plt.show()

# Корреляционный анализ
numeric_cols = X.select_dtypes(include=np.number).columns
corr_matrix = X[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Корреляционная тепловая карта")
plt.show()

# по тепловой карте видно, что наибольшую корреляцию имеют столбики Energy и Loundess, равную 0.76

# Корреляция колонок с таргетом
target_corr = X[numeric_cols].corrwith(y).sort_values()
print("\nКорреляция колонок с таргетом:\n", target_corr)
plt.figure(figsize=(8,6))
target_corr.plot(kind="barh")
plt.title('Корреляция колонок с таргетом')
plt.show()

# Корреляция новых колонок с таргетом
new_feat_corr = X[["duration_min", "loudness_scaled", "tempo_scaled"]].corrwith(y).sort_values()
print("\nКорреляция новых колонок с таргетом:\n", new_feat_corr)
plt.figure(figsize=(8,6))
new_feat_corr.plot(kind='barh')
plt.title('Корреляция новых колонок с таргетом')
plt.show()


# Feature Engineering
categorical_cols = ['artists', 'album_name', 'track_name', 'track_genre']
numeric_cols = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_min', 'loudness_scaled', 'tempo_scaled']


# Создаем трансформер для обработки категориальных признаков
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Random Forest
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=45, min_samples_split=15, min_samples_leaf=2, random_state=42))])
pipeline_rf.fit(X_train, y_train)
rf_pred = pipeline_rf.predict(X_test)
y_test_binary = (y_test > 0).astype(int)
rf_roc_auc = roc_auc_score(y_test_binary, rf_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_recall = recall_score(y_test, rf_pred, average='weighted', zero_division=0)

print("\nRandom Forest метрики:")
print(f"ROC AUC: {rf_roc_auc:.2f}")
print(f"Accuracy: {rf_accuracy:.2f}")
print(f"Precision: {rf_precision:.2f}")
print(f"Recall: {rf_recall:.2f}")

# Логистическая регрессия
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(random_state=42, max_iter=1000))])
pipeline_lr.fit(X_train, y_train)
lr_pred = pipeline_lr.predict(X_test)
y_test_binary = (y_test > 0).astype(int)
lr_roc_auc = roc_auc_score(y_test_binary, lr_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, average='weighted', zero_division=0)
lr_recall = recall_score(y_test, lr_pred, average='weighted', zero_division=0)

print("\nЛогистическая регрессия метрики:")
print(f"ROC AUC: {lr_roc_auc:.2f}")
print(f"Accuracy: {lr_accuracy:.2f}")
print(f"Precision: {lr_precision:.2f}")
print(f"Recall: {lr_recall:.2f}")

# Дерево решений
pipeline_dt = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', DecisionTreeClassifier(random_state=42))])
pipeline_dt.fit(X_train, y_train)
dt_pred = pipeline_dt.predict(X_test)
y_test_binary = (y_test > 0).astype(int)
dt_roc_auc = roc_auc_score(y_test_binary, dt_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred, average='weighted', zero_division=0)
dt_recall = recall_score(y_test, dt_pred, average='weighted', zero_division=0)

print("\nДерево решений метрики:")
print(f"ROC AUC: {dt_roc_auc:.2f}")
print(f"Accuracy: {dt_accuracy:.2f}")
print(f"Precision: {dt_precision:.2f}")
print(f"Recall: {dt_recall:.2f}")

# Нейронная сеть
pipeline_nn = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500, early_stopping=True))])
pipeline_nn.fit(X_train, y_train)
nn_pred = pipeline_nn.predict(X_test)
y_test_binary = (y_test > 0).astype(int)
nn_roc_auc = roc_auc_score(y_test_binary, nn_pred)
nn_accuracy = accuracy_score(y_test, nn_pred)
nn_precision = precision_score(y_test, nn_pred, average='weighted', zero_division=0)
nn_recall = recall_score(y_test, nn_pred, average='weighted', zero_division=0)
print("\nНейронная сеть метрики:")
print(f"ROC AUC: {nn_roc_auc:.2f}")
print(f"Accuracy: {nn_accuracy:.2f}")
print(f"Precision: {nn_precision:.2f}")
print(f"Recall: {nn_recall:.2f}")

"""
результаты
#Random Forest метрики:
ROC AUC: 0.50
Accuracy: 0.14
Precision: 0.06
Recall: 0.14

Логистическая регрессия метрики:
ROC AUC: 0.89
Accuracy: 0.25
Precision: 0.24
Recall: 0.25

Дерево решений метрики:
ROC AUC: 0.75
Accuracy: 0.14
Precision: 0.14
Recall: 0.14

Нейронная сеть метрики:
ROC AUC: 0.95
Accuracy: 0.25
Precision: 0.27
Recall: 0.25
таким образом, самая выгодная метрика: Нейронная сеть
"""