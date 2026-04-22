import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

datos = {
    'size': [50, 60, 80, 100, 120, 150, 200, 250, 300, 350],
    'bedrooms': [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
    'age': [5, 10, 15, 2, 8, 20, 1, 12, 3, 7],
    'price': [150000, 180000, 210000, 250000, 280000, 320000, 450000, 500000, 600000, 650000]
}

df = pd.DataFrame(datos)
print("--- Dataset Original ---")
print(df)
print("\n")

X = df[['size', 'bedrooms', 'age']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Datos para entrenamiento: {len(X_train)} casas")
print(f"Datos para prueba: {len(X_test)} casas\n")

modelo = LinearRegression()
modelo.fit(X_train, y_train)

print("¡El modelo ha sido entrenado!\n")

print("--- Parámetros del Modelo Matemático ---")
print("Coeficientes (peso de cada característica):", modelo.coef_)
print("Intercepto (valor base de una casa):", modelo.intercept_)
print("\n")

predicciones_test = modelo.predict(X_test)
mse = mean_squared_error(y_test, predicciones_test)
r2 = r2_score(y_test, predicciones_test)

print("--- Evaluación del Modelo ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Score R²: {r2:.4f}")
print("\n")

casas_nuevas = pd.DataFrame({
    'size': [85, 220],
    'bedrooms': [2, 4],
    'age': [5, 2]
})

precios_predichos = modelo.predict(casas_nuevas)

print("--- Predicción de Nuevas Casas ---")
for i in range(len(casas_nuevas)):
    print(f"Casa {i+1} (Size: {casas_nuevas.iloc[i]['size']}m², Bedrooms: {casas_nuevas.iloc[i]['bedrooms']}, Age: {casas_nuevas.iloc[i]['age']} años) -> Precio Estimado: ${precios_predichos[i]:,.2f}")
