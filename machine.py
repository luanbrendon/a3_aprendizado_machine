import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados
df = pd.read_csv('diabetes_pt.csv')

# Separar as variáveis preditoras e a variável meta
X = df.drop(columns=['Resultado'])
y = df['Resultado']

# Dividir o conjunto de dados em 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Algoritmo 1: Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
print(f'Acurácia do Random Forest: {accuracy_rf:.2f}%')

# Algoritmo 2: Regressão Logística
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr) * 100
print(f'Acurácia da Regressão Logística: {accuracy_lr:.2f}%')

# Algoritmo 3: K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
print(f'Acurácia do K-Nearest Neighbors: {accuracy_knn:.2f}%')

# Algoritmo 4: Support Vector Machine (SVM)
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm) * 100
print(f'Acurácia do Support Vector Machine: {accuracy_svm:.2f}%')

# Algoritmo 5: Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt) * 100
print(f'Acurácia do Decision Tree: {accuracy_dt:.2f}%')

# Comparando os resultados
print("Resumo dos modelos (em porcentagem):")
print(f"Random Forest: {accuracy_rf:.2f}%")
print(f"Regressão Logística: {accuracy_lr:.2f}%")
print(f"K-Nearest Neighbors: {accuracy_knn:.2f}%")
print(f"Support Vector Machine: {accuracy_svm:.2f}%")
print(f"Decision Tree: {accuracy_dt:.2f}%")
