from sklearn.cluster import KMeans
from typing import List, Tuple
from datetime import datetime, timedelta
import numpy as np
import csv
import sys
import threading
import random
import select
import time

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
    oneClusterBSF, oneClusterLSF, twoClusters1BSF, twoClusters1LSF, twoClusters2BSF, twoClusters2LSF, threeClusters1BSF, threeClusters1LSF, threeClusters2BSF, threeClusters2LSF, threeClusters3BSF, threeClusters3LSF, fourClusters1BSF, fourClusters1LSF, fourClusters2BSF, fourClusters2LSF, fourClusters3BSF, fourClusters3LSF, fourClusters4BSF, fourClusters4LSF = 0
    totalDistance = 0
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
            oneClusterBSF, oneClusterLSF = find_best_path(oneCluster_1)
            totalDistance = oneClusterLSF
        elif i==2:
            twoClusters_1 = clusters[0]
            twoClusters_2 = clusters[1]
            twoClusters1BSF, twoClusters1LSF = find_best_path(twoClusters_1)
            twoClusters2BSF, twoClusters2LSF = find_best_path(twoClusters_2)
            totalDistance = twoClusters1LSF + twoClusters2LSF
        elif i==3:
            threeClusters_1 = clusters[0]
            threeClusters_2 = clusters[1]
            threeClusters_3 = clusters[2]
            threeClusters1BSF, threeClusters1LSF = find_best_path(threeClusters_1)
            threeClusters2BSF, threeClusters2LSF = find_best_path(threeClusters_2)
            threeClusters3BSF, threeClusters3LSF = find_best_path(threeClusters_3)
            totalDistance = threeClusters1LSF + threeClusters2LSF + threeClusters3LSF
        elif i==4:
            fourClusters_1 = clusters[0]
            fourClusters_2 = clusters[1]
            fourClusters_3 = clusters[2]
            fourClusters_4 = clusters[3]
            fourClusters1BSF, fourClusters1LSF = find_best_path(fourClusters_1)
            fourClusters2BSF, fourClusters2LSF = find_best_path(fourClusters_2)
            fourClusters3BSF, fourClusters3LSF = find_best_path(fourClusters_3)
            fourClusters4BSF, fourClusters4LSF = find_best_path(fourClusters_4)
            totalDistance = fourClusters1LSF + fourClusters2LSF + fourClusters3LSF + fourClusters4LSF
        
        print(str(i) + ") If you use " + str(i) + " drone(s), the total route will be " + str(totalDistance) + " meters")
        for k in range(1, i):
            print("    " + str(k) + "Landing Pad " + str(k) + " should be at ")

    
def distance_matrix(coordinates):
    coordinates = np.array(coordinates)
    x = coordinates[:,0]
    y = coordinates[:,1]

    dx = x[:, np.newaxis] - x [np.newaxis, :]
    dy = y[:,np.newaxis] - y[np.newaxis,:]
    dist_matrix = np.sqrt(dx**2 + dy**2)
    return dist_matrix

def nn_temperature(best_path, matrix):
    starting_node = 0

    path = [starting_node]
    visited = {starting_node}
    path_length = 0

    next_node = random.choice([i for i in range(1,len(matrix))])
    path.append(next_node)
    visited.add(next_node)
    path_length = matrix[starting_node][next_node]

    for i in range(len(matrix) - 2):
        curr_node = path[-1]
        next_node = None
        best_distance = float('inf')
        
        for j in range(len(matrix)):
            if j in visited:
                continue
            curr_distance = matrix[curr_node][j]

         
            if curr_distance < best_distance:
                next_node = j
                best_distance = curr_distance
    
        if next_node is None:
            return float('inf'), None
        path_length += best_distance
        path.append(next_node)
        visited.add(next_node)
        
        if path_length >= best_path:
            return float('inf'), None


    path_length += matrix[path[-1]][0]
    path.append(0)

    return path_length, path

def find_best_path(clusterCoords):
    dist_matrix = distance_matrix(clusterCoords)

    best_length = float('inf')
    best_path = None

    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()

    startTime = time.time()

    while (time.time() - startTime) < 10:
        path_length, new_path = nn_temperature(best_length, dist_matrix)

        if path_length < best_length:
            best_length = path_length
            best_path = new_path

    return best_path, best_length

if __name__ == "__main__":
    fileName = input("Enter the name of the file: ")
    tempFileName = "../Dataset/" + fileName
    coords = read_coords_as_tuple(tempFileName)
    estimatedSolutionTime = datetime.now() + timedelta(minutes=5)
    print("There are " + str(len(coords)) + "nodes: Solutions will be available by " + estimatedSolutionTime.strftime("%I:%M %p"))

    createCluster(coords)