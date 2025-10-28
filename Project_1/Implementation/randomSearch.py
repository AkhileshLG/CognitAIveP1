import csv
from typing import List, Tuple
import numpy as np
import random
import msvcrt

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
    path = list(range(n))
    random.shuffle(path)
    path.append(path[0])
    dist = sum(coordinates[path[i]][path[i+1]] for i in range(n))
    return dist, path

def find_best_rand_path(file_path):
    interrupted = False
    coords = read_coords_as_tuple(file_path)
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
    return best_distance, best_path

if __name__ == "__main__":
    file_name=input("file path")
    best_dist, best_path = find_best_rand_path(file_name)
    print(f"Best path found {best_dist}")
    print(f"Best path {best_path}")