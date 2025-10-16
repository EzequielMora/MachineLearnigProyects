import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from datetime import timedelta
import matplotlib.pyplot as plt

# CARGAR DATOS 
df = pd.read_csv("cordoba_clima_completo.csv", parse_dates=['time'])
df = df.sort_values('time')

#  CREAR FEATURES DE LAG (últimos 7 días) 
for lag in range(1, 8):
    df[f'tmin_lag{lag}'] = df['tmin'].shift(lag)
    df[f'tmax_lag{lag}'] = df['tmax'].shift(lag)
    df[f'llueve_lag{lag}'] = df['llueve'].shift(lag)

df['month'] = df['time'].dt.month
df['dayofyear'] = df['time'].dt.dayofyear
df = df.dropna()
 
features = [f'tmin_lag{i}' for i in range(1,8)] + \
           [f'tmax_lag{i}' for i in range(1,8)] + \
           [f'llueve_lag{i}' for i in range(1,8)] + \
           ['month','dayofyear']

X = df[features]
y_tmin = df['tmin']
y_tmax = df['tmax']
y_rain = df['llueve']

# ENTRENAR MODELOS 
X_train, X_test, y_train, y_test = train_test_split(X, y_tmin, test_size=0.2, shuffle=False)
model_tmin = RandomForestRegressor(n_estimators=200, random_state=42)
model_tmin.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y_tmax, test_size=0.2, shuffle=False)
model_tmax = RandomForestRegressor(n_estimators=200, random_state=42)
model_tmax.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y_rain, test_size=0.2, shuffle=False)
model_rain = RandomForestClassifier(n_estimators=200, random_state=42)
model_rain.fit(X_train, y_train)

# PREDICCIÓN PRÓXIMOS 7 DÍAS
last_days = df.tail(7).copy()
fechas = []
tmin_preds = []
tmax_preds = []
rain_preds = []

for i in range(7):
    X_future = pd.DataFrame({
        **{f'tmin_lag{j+1}': [last_days['tmin'].iloc[-j-1]] for j in range(7)},
        **{f'tmax_lag{j+1}': [last_days['tmax'].iloc[-j-1]] for j in range(7)},
        **{f'llueve_lag{j+1}': [last_days['llueve'].iloc[-j-1]] for j in range(7)},
        'month': [(last_days['time'].iloc[-1] + timedelta(days=i+1)).month],
        'dayofyear': [(last_days['time'].iloc[-1] + timedelta(days=i+1)).dayofyear]
    })
    
    tmin_pred = model_tmin.predict(X_future)[0]
    tmax_pred = model_tmax.predict(X_future)[0]
    lluvia_pred = model_rain.predict(X_future)[0]
    
    fechas.append(last_days['time'].iloc[-1] + timedelta(days=i+1))
    tmin_preds.append(tmin_pred)
    tmax_preds.append(tmax_pred)
    rain_preds.append(lluvia_pred)

    new_row = pd.DataFrame({
        'time': [last_days['time'].iloc[-1] + timedelta(days=i+1)],
        'tmin': [tmin_pred],
        'tmax': [tmax_pred],
        'llueve': [lluvia_pred]
    })
    last_days = pd.concat([last_days, new_row], ignore_index=True)

#  GRÁFICO
plt.figure(figsize=(10,5))
plt.plot(fechas, tmin_preds, label='Tmin (°C)', marker='o', color='blue')
plt.plot(fechas, tmax_preds, label='Tmax (°C)', marker='o', color='red')

# Lluvia como barras
rain_colors = ['skyblue' if r==1 else 'white' for r in rain_preds]
plt.bar(fechas, [max(tmax_preds)+1]*7, color=rain_colors, alpha=0.3, label='Lluvia')

plt.title("Predicción Clima Próxima Semana - Córdoba")
plt.xlabel("Fecha")
plt.ylabel("Temperatura (°C)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
