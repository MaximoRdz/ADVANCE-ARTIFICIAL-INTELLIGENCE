import pandas as pd
from river import anomaly
from river import preprocessing

# Cargar el dataset creditcard.csv
file_path = './creditcard.csv'
df = pd.read_csv(file_path)

# Seleccionar las columnas numéricas que serán usadas para el algoritmo
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Inicializar el escalador estándar
scaler = preprocessing.StandardScaler()

# Inicializar el detector de anomalías Half-Space Trees
hst = anomaly.HalfSpaceTrees(n_trees=15)

# Contador de anomalías
anomalies_count = 0

# Ajustar el umbral de anomalía para ser más estricto
anomaly_threshold = 0.98  # Mayor valor --> Menos anomalías

# Iterar sobre cada fila del DataFrame
for i, row in df.iterrows():
    # Crear un diccionario con los valores de la fila para las columnas numéricas
    data_point = {col: row[col] for col in numeric_columns}
    
    # Actualizar el escalador con el punto actual
    scaler.learn_one(data_point)
    
    # Estandarizar el punto de datos
    data_point = scaler.transform_one(data_point)
    
    # Asegúrate de que el modelo ha sido entrenado al menos una vez antes de calcular la puntuación
    if hst.n_trees > 0:
        # Obtener la puntuación de anomalía de HST para el punto actual
        anomaly_score = hst.score_one(data_point)
    else:
        anomaly_score = 0
    
    # Actualizar el modelo de HST con el punto actual
    hst.learn_one(data_point)

    # Si la puntuación de anomalía es mayor que el umbral, marcar como anomalía
    if anomaly_score > anomaly_threshold:
        anomalies_count += 1
        print(f"Anomalía detectada en fila {i} con puntuación {anomaly_score:.2f}")

# Mostrar el total de anomalías detectadas
print(f"Total de anomalías detectadas: {anomalies_count}")

