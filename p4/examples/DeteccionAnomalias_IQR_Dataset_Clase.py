import pandas as pd
from river import cluster, preprocessing, compose
import math

# Cargar el archivo creditcard.csv en un DataFrame usando pandas para simular un flujo de datos
file_path = './creditcard.csv'
df = pd.read_csv(file_path)

# Crear un pipeline con estandarización y KMeans
pipeline = compose.Pipeline(
    ('scaler', preprocessing.StandardScaler()),  # Paso de estandarización
    ('kmeans', cluster.KMeans(n_clusters=2, seed=42))  # Paso de clustering
)

# Umbral para considerar un punto como anomalía basándonos en la distancia al centroide
threshold = 3.0

# Para almacenar el conteo de anomalías
anomalies_detected = 0

# Función para calcular la distancia euclidiana entre un punto y un centroide
def euclidean_distance(point, center):
    return math.sqrt(sum((point[feature] - center[feature]) ** 2 for feature in point))

# Iterar sobre cada fila del DataFrame para simular un flujo de datos
for _, row in df.iterrows():
    # Seleccionar todas las columnas numéricas excepto 'Time' y 'Class'
    features = row.drop(labels=['Time', 'Class']).to_dict()

    # Usar el pipeline para normalizar y luego aplicar K-Means
    cluster_id = pipeline.predict_one(features)

    # Obtener el paso de KMeans desde el pipeline
    kmeans = pipeline['kmeans']  # Acceder al modelo KMeans dentro del pipeline

    # Obtener los centroides aprendidos hasta el momento
    centroids = kmeans.centers

    # Si el modelo ha aprendido suficientes centroides, calcular la distancia
    if centroids:
        # Obtener el centroide del clúster asignado
        centroid = centroids[cluster_id]

        # Calcular la distancia euclidiana al centroide
        distance_to_centroid = euclidean_distance(features, centroid)

        # Si la distancia es mayor que el umbral, lo consideramos una anomalía
        if distance_to_centroid > threshold:
            anomalies_detected += 1
            print(f"Anomalía detectada en el punto: {row['Time']} con distancia {distance_to_centroid:.2f}")

    # Actualizar el pipeline con el nuevo punto
    pipeline = pipeline.learn_one(features)

# Mostrar el número total de anomalías detectadas
print(f"Total de anomalías detectadas: {anomalies_detected}")
