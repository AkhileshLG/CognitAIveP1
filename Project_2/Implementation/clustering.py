from sklearn.cluster import KMeans
from typing import List, Tuple
import numpy as np
import csv

def read_coords_as_tuple(file_path:str) -> List[Tuple[float,float]]: 
    coordinates = []
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row_num, row in enumerate(csv_reader, 1):
            if len(row) != 2:
                raise ValueError(f"Row {row_num} must contain 2 values")
            x, y = map(float, row)
            coordinates.append((x, y))
    return coordinates

def createCluster(coords):
    oneCluster_1, twoClusters_1, twoClusters_2, threeClusters_1, threeClusters_2, threeClusters_3, fourClusters_1, fourClusters_2, fourClusters_3, fourClusters_4 = []

    coords_np = np.array(coords)

    for i in range(1,4):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(coords_np)

        labels = kmeans.labels_

        clusters = []
        for j in range(kmeans.n_clusters):
            cluster_coords = coords_np[labels == j].toList()
            clusters.append(cluster_coords)

            if i==1:
                oneCluster_1 = clusters[0]
            elif i==2:
                twoClusters_1 = clusters[0]
                twoClusters_2 = clusters[1]
            elif i==3:
                threeClusters_1 = clusters[0]
                threeClusters_2 = clusters[1]
                threeClusters_3 = clusters[2]
            elif i==4:
                fourClusters_1 = clusters[0]
                fourClusters_2 = clusters[1]
                fourClusters_3 = clusters[2]
                fourClusters_4 = clusters[3]
    

if __name__ == "__main__":
    fileName = input("Enter the name of the file: ")
    tempFileName = "../Dataset/" + fileName
    coords = read_coords_as_tuple(tempFileName)

    createCluster(coords)