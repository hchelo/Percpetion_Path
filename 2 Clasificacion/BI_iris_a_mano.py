from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import math

# Cargar la base de datos Iris
iris = load_iris()

# Acceder a los datos y etiquetas
X = iris.data  # Características (atributos)
y = iris.target  # Etiquetas (clases)

# Imprimir información sobre la base de datos
print("Información de la base de datos Iris:")
print("Características (atributos):", iris.feature_names)
print("Etiquetas (clases):", iris.target_names)
print("Tamaño de los datos:", X.shape)
print("Tamaño de las etiquetas:", y.shape)

# Imprimir los primeros 5 datos y etiquetas
print("Primeros 5 datos:")
print(X[0:5, :])
print("Etiquetas correspondientes:")
print(y[0:5])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_setosa = X[0:25]
Y_train_setosa = y[0:25]
X_test_setosa = X[25:50]  # Características (atributos)
Y_test_setosa = y[25:50]  # Etiquetas (clases)

X_train_versi = X[50:75]
Y_train_versi = y[50:75]
X_test_versi = X[75:100]  # Características (atributos)
Y_test_versi = y[75:100]  # Etiquetas (clases)

X_train_virgi = X[100:125]
Y_train_virgi = y[100:125]
X_test_virgi = X[125:150]  # Características (atributos)
Y_test_virgi = y[125:150]  # Etiquetas (clases)

# Juntando los datos test en un solo vector
X_Test_total = np.concatenate((X_test_setosa, X_test_versi, X_test_virgi), axis=0)
Y_Test_total = np.concatenate((Y_test_setosa, Y_test_versi, Y_test_virgi), axis=0)

# Función para calcular parámetros
def calcular_parametros(X_train):
    u = np.mean(X_train, axis=0)
    cov = np.diag(np.diag(np.cov(X_train, rowvar=False)))
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    return u, cov, inv_cov, det_cov

# Calcular parámetros para cada clase
u_set, cov_set, inv_cov_set, det_cov_set = calcular_parametros(X_train_setosa)
u_versi, cov_versi, inv_cov_versi, det_cov_versi = calcular_parametros(X_train_versi)
u_virgi, cov_virgi, inv_cov_virgi, det_cov_virgi = calcular_parametros(X_train_virgi)

# Función para calcular la probabilidad gaussiana
def calcular_gaussiana(x, u, inv_cov, det_cov):
    diff = x - u
    exponent = -0.5 * np.dot(np.dot(diff, inv_cov), diff.T)
    denominator = math.sqrt((2 * math.pi) ** len(u) * det_cov)
    return (1 / denominator) * math.exp(exponent)

# Clasificación
Y_result = np.zeros(len(X_Test_total))

for i, x in enumerate(X_Test_total):
    gaussian_set = calcular_gaussiana(x, u_set, inv_cov_set, det_cov_set)
    gaussian_versi = calcular_gaussiana(x, u_versi, inv_cov_versi, det_cov_versi)
    gaussian_virgi = calcular_gaussiana(x, u_virgi, inv_cov_virgi, det_cov_virgi)
    
    if gaussian_set > gaussian_versi and gaussian_set > gaussian_virgi:
        Y_result[i] = 0
    elif gaussian_versi > gaussian_set and gaussian_versi > gaussian_virgi:
        Y_result[i] = 1
    else:
        Y_result[i] = 2

print("Etiquetas de prueba:")
print(Y_Test_total)
print("Etiquetas predichas:")
print(Y_result)

# Construir la matriz de confusión
confusion = confusion_matrix(Y_Test_total, Y_result)
print("Matriz de confusión:")
print(confusion)
