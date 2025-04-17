import pandas as pd
from river import cluster, preprocessing
import math

# Cargar el archivo creditcard.csv en un DataFrame usando pandas
file_path = './creditcard.csv'
df = pd.read_csv(file_path)

# Crear un scaler para estandarizar los datos
scaler = preprocessing.StandardScaler()

# Inicializar K-Means con 2 clústeres (puedes ajustar este valor según tus datos)
kmeans = cluster.KMeans(n_clusters=5, seed=42)

# Umbral para considerar un punto como anomalía basándonos en la distancia al centroide
threshold = 60.0

# Para almacenar el conteo de anomalías
anomalies_detected = 0

# Función para calcular la distancia euclidiana entre un punto y un centroide
def euclidean_distance(point, center):
    return math.sqrt(sum((point[feature] - center[feature]) ** 2 for feature in point))

# Iterar sobre cada fila del DataFrame para simular un flujo de datos
for _, row in df.iterrows():
    # Seleccionar todas las columnas numéricas excepto 'Time' y 'Class'
    features = row.drop(labels=['Time', 'Class']).to_dict()

    # Actualizar el escalador con los nuevos datos
    scaler.learn_one(features)
    
    # Estandarizar las características
    features = scaler.transform_one(features)

    # Predecir el clúster más cercano
    cluster_id = kmeans.predict_one(features)

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

    # Actualizar el modelo KMeans con el nuevo punto
    kmeans.learn_one(features)

# Mostrar el número total de anomalías detectadas
print(f"Total de anomalías detectadas: {anomalies_detected}")

