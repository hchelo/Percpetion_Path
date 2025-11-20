from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Cargar la base de datos Iris
iris = load_iris()

# Acceder a los datos y etiquetas
X = iris.data  # Características (atributos)
y = iris.target  # Etiquetas (clases)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Crear el modelo de Naive Bayes
gnb = GaussianNB()

# Entrenar el modelo
gnb.fit(X_train, y_train)

# Realizar predicciones
y_pred = gnb.predict(X_test)

# Imprimir resultados
print("Etiquetas de prueba:")
print(y_test)
print("Etiquetas predichas:")
print(y_pred)

# Construir la matriz de confusión
confusion = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(confusion)

# Calcular y mostrar la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión: {accuracy:.2f}")
