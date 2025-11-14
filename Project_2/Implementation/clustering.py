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
import matplotlib.pyplot as plt

def plot_clusters_and_paths(coords, labels, center, cluster_best_paths, output = "clusters"):
    plt.figure(figsize=(8,8))
    num_clusters = len(center)
    cmap = plt.get_cmap("tab10",num_clusters)
    coords = np.array(coords)

    for i in range(num_clusters):
        mask = (labels ==i)
        clusters_points = coords[mask]
        color = cmap (i%10)
        if clusters_points.size > 0:
            plt.scatter(clusters_points[:,0], clusters_points[:,1], color = color, s = 20, label = f"Cluster {i+1}")
        c = center[i]
        plt.scatter(c[0],c[1], marker = 'X', s = 100, color = color, edgecolor = 'blue')
        if i < len(cluster_best_paths):
            path = cluster_best_paths[i]
            cluster_coordinates = clusters_points.tolist()
            try:
                path_coords = [cluster_coordinates[idx] for idx in path]
                xs = [p[0] for p in path_coords]
                ys = [p[1] for p in path_coords]
                plt.plot(xs,ys, '-', color = color, linewidth=1)
                plt.plot(xs,ys, 'o', color = color, markersize=4)
            except Exception:
                pass
    plt.title(output)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc = "best")
    plt.savefig(output + ".png", dpi = 100)
    plt.close()

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

def createCluster(coords, file_name):
    coords_np = np.array(coords)
    oneClusterBSF = []
    oneClusterLSF = []
    twoClusterBSF = []
    twoClusterLSF = []
    threeClusterBSF = []
    threeClusterLSF = []
    fourClusterBSF = []
    fourClusterLSF = []
    oneCentroid = []
    twoCentroids = []
    threeCentroids = []
    fourCentroids = []
    oneLabels = []
    twoLabels = []
    threeLabels = []
    fourLabels = []
    
    for i in range(1,5):
        kmeans = KMeans(n_clusters=i, n_init="auto")
        kmeans.fit(coords_np)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        clusters = []
        individualDists = []

        for j in range(i):
            clusters_coords = coords_np[labels == j].tolist()
            clusters.append(clusters_coords)
            
        totalDistance = 0
        cluster_best_paths = []
        for cluster in clusters:
            if len(cluster) > 0:
                best_path, clusterDist = find_best_path(cluster)
                totalDistance += clusterDist
                cluster_best_paths.append(best_path)
                individualDists.append(clusterDist)
               
                if i==1:
                    oneClusterBSF.append(best_path)
                    oneClusterLSF.append(clusterDist)
                    oneCentroid = centroids
                    oneLabels = labels
                elif i==2:
                    twoClusterBSF.append(best_path)
                    twoClusterLSF.append(clusterDist)
                    twoCentroids = centroids
                    twoLabels = labels
                elif i==3:
                    threeClusterBSF.append(best_path)
                    threeClusterLSF.append(clusterDist)
                    threeCentroids = centroids
                    threeLabels = labels
                elif i==4:
                    fourClusterBSF.append(best_path)
                    fourClusterLSF.append(clusterDist)
                    fourCentroids = centroids
                    fourLabels = labels
            else:
                cluster_best_paths.append(None)
    
        print(f"{i}) If you use {i} drone(s), the total route will be {totalDistance:.2f} meters")
        for j, c in enumerate(centroids):
            print(f"    Landing Pad {j+1} should be at ({c[0]:.2f}, {c[1]:.2f}), serving {len(clusters[j])}, route is {individualDists[j]:.2f}")
        
    output = f"{file_name}_{i}_Drones"    
    
    solution1 = ""
    solution2 = ""
    solution3 = ""
    solution4 = ""
    fileNumber = int(input("Please select your choice 1 to 4: "))
    if fileNumber == 1:
        solution1 = fileName + "_1_SOLUTION_" + str(round(oneClusterLSF[0])) + ".txt"
        print("Writing " + solution1 + " to disk")
        plot_clusters_and_paths(coords_np, oneLabels, oneCentroid, oneClusterBSF, output=output)
        with open(solution1, "a") as f:
            f.write(oneClusterBSF[0])
    elif fileNumber == 2:
        solution1 = fileName + "_1_SOLUTION_" + str(round(twoClusterLSF[0])) + ".txt"
        solution2 = fileName + "_2_SOLUTION_" + str(round(twoClusterLSF[1])) + ".txt"
        print("Writing " + solution1 + ", " + solution2 + " to disk")
        plot_clusters_and_paths(coords_np, twoLabels, twoCentroids, twoClusterBSF, output=output)
        with open(solution1, "a") as f:
            f.write(twoClusterBSF[0])
        with open(solution2, "a") as f:
            f.write(twoClusterBSF[1])
    elif fileNumber == 3:
        solution1 = fileName + "_1_SOLUTION_" + str(round(threeClusterLSF[0])) + ".txt"
        solution2 = fileName + "_2_SOLUTION_" + str(round(threeClusterLSF[1])) + ".txt"
        solution3 = fileName + "_3_SOLUTION_" + str(round(threeClusterLSF[2])) + ".txt"
        print("Writing " + solution1 + ", " + solution2 + ", " + solution3 + " to disk")
        plot_clusters_and_paths(coords_np, threeLabels, threeCentroids, threeClusterBSF, output=output)
        with open(solution1, "a") as f:
            f.write(threeClusterBSF[0])
        with open(solution2, "a") as f:
            f.write(threeClusterBSF[1])
        with open(solution3, "a") as f:
            f.write(threeClusterBSF[2])
    elif fileNumber == 4:
        solution1 = fileName + "_1_SOLUTION_" + str(round(fourClusterLSF[0])) + ".txt"
        solution2 = fileName + "_2_SOLUTION_" + str(round(fourClusterLSF[1])) + ".txt"
        solution3 = fileName + "_3_SOLUTION_" + str(round(fourClusterLSF[2])) + ".txt"
        solution4 = fileName + "_4_SOLUTION_" + str(round(fourClusterLSF[3])) + ".txt"
        print("Writing " + solution1 + ", " + solution2 + ", " + solution3 + ", " + solution4 + " to disk")
        plot_clusters_and_paths(coords_np, fourLabels, fourCentroids, fourClusterBSF, output=output)
        with open(solution1, "a") as f:
            f.write(str(fourClusterBSF[0]))
        with open(solution2, "a") as f:
            f.write(str(fourClusterBSF[1]))
        with open(solution3, "a") as f:
            f.write(str(fourClusterBSF[2]))
        with open(solution4, "a") as f:
            f.write(str(fourClusterBSF[3]))


def distance_matrix(coordinates):
    coordinates = np.array(coordinates)
    x = coordinates[:,0]
    y = coordinates[:,1]
    dx = x[:, np.newaxis] - x[np.newaxis, :]
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
    
    startTime = time.time()
    while (time.time() - startTime) < 25:
        path_length, new_path = nn_temperature(best_length, dist_matrix)
        if path_length < best_length:
            best_length = path_length
            best_path = new_path

    return best_path, best_length

if __name__ == "__main__":
    fileName = input("Enter the name of the file: ")
    tempFileName = "../Dataset/" + fileName + ".csv"
    coords = read_coords_as_tuple(tempFileName)
    estimatedSolutionTime = datetime.now() + timedelta(minutes=5)
    print("There are " + str(len(coords)) + " nodes: Solutions will be available by " + estimatedSolutionTime.strftime("%I:%M %p"))
    
    createCluster(coords, fileName)