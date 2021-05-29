
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD


datos = pd.read_csv('dataset3.csv', sep=",", skiprows=0, usecols=[4,474])
print(datos.dtypes)

# Al graficar los datos se observa una tendencia lineal
datos.plot.scatter(x='total', y='poblacion')
plt.xlabel('Total Muertes')
plt.ylabel('poblacion')
plt.show()

x = datos['total'].values
y = datos['poblacion'].values

#
# Construir el modelo en Keras
#

# - Capa de entrada: 1 dato (cada dato "x" correspondiente a la edad)
# - Capa de salida: 1 dato (cada dato "y" correspondiente a la regresión lineal)
# - Activación: 'linear' (pues se está implementando la regresión lineal)

np.random.seed(2)			# Para reproducibilidad del entrenamiento

input_dim = 1
output_dim = 1000
modelo = Sequential()
modelo.add(Dense(output_dim, input_dim=input_dim, activation='linear'))
modelo.add(Dense(8, activation='linear'))
modelo.add(Dense(1, activation='linear'))
#modelo.add(Dense(1, input_dim=input_dim, activation='linear'))
# Definición del método de optimización (gradiente descendiente), con una
# tasa de aprendizaje de 0.0004 y una pérdida igual al error cuadrático
# medio

sgd = SGD(lr=0.0004)
#modelo.compile(loss='mse', optimizer=sgd)
modelo.compile(loss='mean_squared_error',
        optimizer='adam',
        metrics= ['binary_accuracy'])

# Imprimir en pantalla la información del modelo
modelo.summary()

#
# Entrenamiento: realizar la regresión lineal
#

# 40000 iteraciones y todos los datos de entrenamiento (29) se usarán en cada
# iteración (batch_size = 29)

num_epochs = 20000
batch_size = x.shape[0]
history = modelo.fit(y, x, epochs=num_epochs, batch_size=batch_size, verbose=0)
#
# Visualizar resultados del entrenamiento
#

# Imprimir los coeficientes "w" y "b"
capas = modelo.layers[0]
w, b = capas.get_weights()
print('Parámetros: w = {:.1f}, b = {:.1f}'.format(w[0][0],b[0]))

# Graficar el error vs epochs y el resultado de la regresión
# superpuesto a los datos originales
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('ECM')
plt.title('ECM vs. epochs')

x_regr = modelo.predict(y)
plt.subplot(1, 2, 2)
plt.scatter(y,x)
plt.plot(y,x_regr,'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Datos originales y regresión lineal')
plt.show()

# Predicción
x_pred = np.array([30000])
y_pred = modelo.predict(x_pred)
print("Habrán {:.1f} muertos".format(y_pred[0][0]), " para una poblacion de {} personas".format(x_pred[0]))