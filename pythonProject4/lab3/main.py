import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearModel:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.zeros(self.num_features)
        self.bias = 0

    def __call__(self, X):
        return np.dot(X,self.weights) + self.bias

class LinearRegressor(LinearModel):
    def fit(self, X, y, learning_rate=0.01, epochs=100):
        error_history = []
        for _ in range(epochs):
            predictions = self.predict(X)
            error = y - predictions
            gradient = (2 * np.dot(X.T, error)) / len(X)
            self.weights -= learning_rate * gradient
            self.bias -= learning_rate * np.mean(error)
            current_error = ((y - predictions) ** 2).sum()
            error_history.append(current_error)
        return error_history

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class LinearClassifier(LinearModel):
    def fit(self, X, y, learning_rate=0.01, epochs=100):
        error_history = []
        for _ in range(epochs):
            predictions = self.predict_proba(X)
            sigmoid = 1 / (1 + np.exp(-np.dot(X, self.weights) - self.bias))
            error = y - sigmoid
            gradient = (2 * np.dot(X.T, error)) / len(X)
            self.weights -= learning_rate * gradient
            self.bias -= learning_rate * np.mean(error)
            current_error = -np.mean(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid))
            error_history.append(current_error)
        return error_history

    def predict(self, X):
        predictions = np.dot(X, self.weights) + self.bias
        sigmoid = 1 / (1 + np.exp(-predictions))
        return np.round(sigmoid)

    def predict_proba(self, X):
        predictions = np.dot(X, self.weights) + self.bias
        sigmoid = 1 / (1 + np.exp(-predictions))
        return sigmoid





data = pd.read_csv('german.csv', sep=';')

Y = data['Creditability']
X = data[['Account_Balance', 'Payment_Status_of_Previous_Credit', 'Value_Savings_Stocks', 'Length_of_current_employment', 'Sex_Marital_Status', 'No_of_Credits_at_this_Bank', 'Guarantors', 'Concurrent_Credits', 'Purpose']]
mean = np.mean(X)
std = np.std(X)
normalized_X = (X - mean) / std
lc = LinearClassifier(num_features=X.shape[1])
history = lc.fit(normalized_X, Y)
epochs = range(1, len(history) + 1)



pred = lc.predict(normalized_X)
print('accuracy: ', (pred == Y).sum() / len(Y))


a = np.round(lc.predict_proba(X))
tp = np.sum((Y == 1) & (a == 1))
fp = np.sum((Y == 0) & (a == 1))
fn = np.sum((Y == 1) & (a == 0))
precision = 0
recall = 0
f1 = 0
if (tp + fp) > 0:
    precision = tp/ (tp+fp)
if (tp + fn) > 0:
    recall = tp / (tp+fn)
if (precision + recall) > 0:
    f1 = 2 * (precision * recall) / (precision + recall)


a = lc.predict_proba(X)

n_positive = np.sum(Y)
n_negative = len(Y) - n_positive
sorted_idx = np.argsort(a)[::-1]
y_true_sorted = Y[sorted_idx]

tpr = np.cumsum(y_true_sorted) / n_positive
fpr = np.cumsum(1 - y_true_sorted) / n_negative
auc = np.trapz(tpr, fpr)

print(precision,recall,f1, auc)





