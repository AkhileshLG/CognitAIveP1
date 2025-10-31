import csv
from typing import List, Tuple
import numpy as np
import random
import msvcrt
import math
import matplotlib.pyplot as plt

def plotAndSavePath(coords, best_path, fileName):
    x = [float(coords[i][0]) for i in best_path]
    y = [float(coords[i][1]) for i in best_path]

    plt.plot(x, y, '-o', color='blue', markersize=6)
    plt.scatter(x[0], y[0], color='orange', s=100, zorder=5)
    
    for i, (x_i, y_i) in enumerate(coords):
        plt.text(x_i + 0.1, y_i + 0.1, str(i), fontsize=10)
    
    plt.title(fileName + " Path")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")

    plt.savefig(fileName + "_path.jpg", format='jpg', dpi=300)
    return

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

def distance_matrix(coordinates):
    coordinates = np.array(coordinates)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    return np.sqrt(dx**2 + dy**2)

def random_path(coordinates):
    n = len(coordinates)
    middle = list(range(1, n))
    random.shuffle(middle)
    path = [0] + middle + [0]
    dist = sum(coordinates[path[i]][path[i+1]] for i in range(n))
    return dist, path

def find_best_rand_path(file_path):
    coords = read_coords_as_tuple(file_path)

    if len(coords) <= 0:
        return 0, True
    elif len(coords) > 256:
        return 0, False
    else:
        print("There are " + str(len(coords)) + " nodes, computing route..")
        print("  Shortest Route Discovered So Far")
    dist_matrix = distance_matrix(coords)
    best_distance = float('inf')
    best_path = None

    for i in range(0, math.factorial(len(coords) - 1)):
        if msvcrt.kbhit() and msvcrt.getch() == b"\r":
            break
        distance,path = random_path(dist_matrix)
        if distance < best_distance:
            best_distance = distance
            best_path = path
            print("    " + str(best_distance))
    return best_distance, best_path, coords

if __name__ == "__main__":
    file_name=input("Enter the name of file: ")
    tempFileName = "../Dataset_csv/" + file_name
    best_dist, best_path, coords = find_best_rand_path(tempFileName)

    if best_path == True:
        print("There are less that 1 node, resulting in no solution")
    elif best_path == False:
        print("There are more than 256 nodes, resulting in no solution")
    else:
        newFileName = file_name[:-4]
        newFileName = newFileName + "_solution"

        with open(newFileName + ".txt", "w") as f:
            print("Route written to disk as " + str(newFileName) + ".txt")
            for i in range(0, len(best_path)):
                if i < len(best_path):
                    f.write(str(best_path[i]) + " ")
                else:
                    f.write(str(best_path[i]))
        
        plotAndSavePath(coords, best_path, newFileName)