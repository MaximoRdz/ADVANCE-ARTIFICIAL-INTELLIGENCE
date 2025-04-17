import pandas as pd
from river import stats

# Se carga el dataset que se va a utilizar:
file_path = './creditcard.csv'
df = pd.read_csv(file_path)

# Crear diccionario para almacenar los cuantiles (Q1 y Q3) para cada característica numérica
iqr_stats = {col: (stats.Quantile(0.25), stats.Quantile(0.75)) for col in df.select_dtypes(include=['float64', 'int64']).columns}

# Limitar el rango intercuartil: el rango "normal" suele considerarse dentro de 1.5 veces el IQR
iqr_factor = 1.5

# Inicializar un diccionario para contar las anomalías detectadas
anomalies = {col: 0 for col in iqr_stats.keys()}

# Función para verificar si un punto es una anomalía basado en IQR
def is_anomaly(val, q1, q3):
    iqr = q3 - q1
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    return val < lower_bound or val > upper_bound

# Iterar sobre cada fila de los datos como si fuera un flujo de datos
for _, row in df.iterrows():
    for col, (q1_stat, q3_stat) in iqr_stats.items():
        val = row[col]

        # Actualizar los cuantiles de Q1 y Q3 de manera incremental
        q1_stat.update(val)
        q3_stat.update(val)

        # Obtener los valores actuales de Q1 y Q3
        q1 = q1_stat.get()
        q3 = q3_stat.get()

        # Comprobar que tanto q1 como q3 no son None (es decir, que tienen suficientes datos)
        if q1 is not None and q3 is not None:
            # Verificar si el valor es una anomalía
            if is_anomaly(val, q1, q3):
                anomalies[col] += 1
                print(f"Anomalía detectada en {col} con valor {val}")

# Mostrar el número total de anomalías detectadas por columna
print("Anomalías detectadas por columna:")
print(anomalies)