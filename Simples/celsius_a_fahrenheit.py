import numpy as np
from sklearn.linear_model import LinearRegression
#modelo siemple de ML para sacar formula de conversion de grados celsius a fahrenheit
#Esta es muy simple, ya que es una regrecion lineal y con 2 datos sin ruido ya es exacta
#pero me sirve para ver las funciones de sklearn

X_celsius = np.array([-40, -10], dtype=float).reshape(-1, 1)
y_fahrenheit = np.array([-40, 14], dtype=float)

modelo = LinearRegression()

print("Entrenando el modelo con los valores manuales de X e Y...")
modelo.fit(X_celsius, y_fahrenheit)
print("¡Entrenamiento completado!\n")


m = modelo.coef_[0]
b = modelo.intercept_

print("El modelo ha 'descubierto' la siguiente fórmula:")
print(f"Fahrenheit = ({m:.2f} * Celsius) + {b:.2f}")
print("(Nota: Esto es idéntico a la fórmula real de conversión)\n")


valor_nuevo = np.array([[100]]) 
resultado = modelo.predict(valor_nuevo)

print(f"Para 100°C, el modelo predice: {resultado[0]:.2f}°F")
