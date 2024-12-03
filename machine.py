import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('diabetes_pt.csv')

X = df.drop(columns=['Resultado'])
y = df['Resultado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_names = []
accuracies = []

def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    model_names.append(model_name)
    accuracies.append(accuracy)

rf_model = RandomForestClassifier(random_state=42)
evaluate_model(rf_model, "Random Forest")

lr_model = LogisticRegression(max_iter=1000, random_state=42)
evaluate_model(lr_model, "Regressão Logística")

knn_model = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn_model, "K-Nearest Neighbors")

svm_model = SVC(random_state=42)
evaluate_model(svm_model, "Support Vector Machine")

dt_model = DecisionTreeClassifier(random_state=42)
evaluate_model(dt_model, "Decision Tree")

results_df = pd.DataFrame({
    'Modelo': model_names,
    'Acurácia (%)': accuracies
})

print("\nComparação de Acurácia entre Modelos:")
print(results_df)

plt.figure(figsize=(10, 6))
bars = plt.barh(results_df['Modelo'], results_df['Acurácia (%)'], color='skyblue')

for bar in bars:
    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}%', 
             va='center', ha='left', fontsize=10, color='black')

plt.xlabel('Acurácia (%)')
plt.title('Comparação de Acurácia entre Algoritmos')
plt.xlim(0, 100)
plt.show()