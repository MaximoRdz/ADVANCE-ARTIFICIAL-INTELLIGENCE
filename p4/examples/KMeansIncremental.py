from river import cluster
from river import metrics
from river.datasets import synth

# Cargamos un conjunto de datos de flujo
datasetSinteticoSEA = synth.SEA()
dataset = datasetSinteticoSEA.take(1000)

# Definimos un modelo K-Means Incremental con 3 clusters
model = cluster.KMeans(n_clusters=3)

# Métrica de evaluación: medida de silhouette para clustering
metric = metrics.Silhouette()

# Diccionario para contar el número de ejemplos en cada cluster
cluster_counts = {i: 0 for i in range(model.n_clusters)}

# Proceso de clustering incremental
for x, _ in dataset:
    model.learn_one(x)  # Aprende de un nuevo dato
    y_pred = model.predict_one(x)  # Asigna el nuevo dato a un cluster
    cluster_counts[y_pred] += 1  # Incrementa el contador para el cluster predicho
    metric.update(x, y_pred, model.centers)  # Actualiza la métrica de Silhouette
    
    # Imprimir el cluster asignado para cada dato
    # print(f"Ejemplo: {x} asignado al cluster: {y_pred}")

# Mostrar resultados finales
print("\nResultados finales:")
print(f"Silhouette score final: {metric.get()}")

# Mostrar los clusters y el número de ejemplos asignados a cada uno
print("\nNúmero de ejemplos en cada cluster:")
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id}: {count} ejemplos")

# Mostrar los centros de los clusters
print("\nCentros de los clusters:")
for cluster_id, center in model.centers.items():
    print(f"Cluster {cluster_id}: Centro {center}")