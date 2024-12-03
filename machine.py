import pandas as pd
import matplotlib.pyplot as plt
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

# Lista para armazenar os resultados de acurácia
model_names = []
accuracies = []

# Função para adicionar acurácia de cada modelo à lista
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    model_names.append(model_name)
    accuracies.append(accuracy)

# Algoritmo 1: Random Forest
rf_model = RandomForestClassifier(random_state=42)
evaluate_model(rf_model, "Random Forest")

# Algoritmo 2: Regressão Logística
lr_model = LogisticRegression(max_iter=1000, random_state=42)
evaluate_model(lr_model, "Regressão Logística")

# Algoritmo 3: K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn_model, "K-Nearest Neighbors")

# Algoritmo 4: Support Vector Machine (SVM)
svm_model = SVC(random_state=42)
evaluate_model(svm_model, "Support Vector Machine")

# Algoritmo 5: Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
evaluate_model(dt_model, "Decision Tree")

# Plotar o gráfico de barras para comparar a acurácia
plt.figure(figsize=(10, 6))
plt.barh(model_names, accuracies, color='skyblue')
plt.xlabel('Acurácia (%)')
plt.title('Comparação de Acurácia entre Algoritmos')
plt.xlim(0, 100)
plt.show()

# Comparando os resultados
print("\nResumo dos modelos (em porcentagem):")
for name, accuracy in zip(model_names, accuracies):
    print(f"{name}: {accuracy:.2f}%")
