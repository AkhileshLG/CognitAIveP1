import csv
from typing import List, Tuple
import numpy as np
import random
import msvcrt

def read_coords_as_tuple(file_path:str) -> List[Tuple[float,float]]: 
    coordinates = []
    numOfNodes = 0
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row_num, row in enumerate(csv_reader, 1):
            if len(row) != 2:
                raise ValueError(f"Row {row_num} must contain 2 values")
            x, y = map(float, row)
            coordinates.append((x, y))
            numOfNodes += 1
    return coordinates, numOfNodes

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
    interrupted = False
    coords, numOfNodes = read_coords_as_tuple(file_path)

    if numOfNodes <= 0:
        return 0, True
    elif numOfNodes > 256:
        return 0, False
    else:
        print("There are " + str(numOfNodes) + "nodes, computing route..")
        print("  Shortest Route Discovered So Far")
    dist_matrix = distance_matrix(coords)
    best_distance = float('inf')
    best_path = None

    for i in range(1, 100000):
        if msvcrt.kbhit() and msvcrt.getch() == b"\r":
            break
        distance,path = random_path(dist_matrix)
        if distance < best_distance:
            best_distance = distance
            best_path = path
            print("    " + str(best_distance))
    return best_distance, best_path

if __name__ == "__main__":
    file_name=input("Enter the name of file: ")
    best_dist, best_path = find_best_rand_path(file_name)

    if best_path == True:
        print("There are less that 1 node, resulting in no solution")
    elif best_path == False:
        print("There are more than 256 nodes, resulting in no solution")
    else:
        print(f"Best path found {best_dist}")
        print(f"Best path {best_path}")