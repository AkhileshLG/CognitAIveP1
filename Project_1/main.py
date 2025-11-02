import csv
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import random
import math

def read_coords_as_tuple(file_path:str) -> List[Tuple[float,float]]: 
    coordinates = []
    try:
        with open(file_path, 'r' , newline = '') as csvfile:
            csv_reader = csv.reader(csvfile)

            for row_num, row in enumerate(csv_reader,1):
                if(len(row)!=2):
                    raise ValueError(f"Row {row_num} must contain 2 values")
                try : 
                    x = float(row[0])
                    y = float(row[1])
                    coordinates.append((x,y))
                except ValueError as e:
                    raise ValueError (f"Row {row_num}: Unable to convert values to float")
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    return coordinates

def distance_matrix(coordinates):
    coordinates = np.array(coordinates)
    x = coordinates[:,0]
    y = coordinates[:,1]

    dx = x[:, np.newaxis] - x [np.newaxis, :]
    dy = y[:,np.newaxis] - y[np.newaxis,:]
    dist_matrix = np.sqrt(dx**2 + dy**2)
    return dist_matrix


def plot_path(coords, path, filename):
    coords = np.array(coords)
    plt.figure(figsize=(7,7))
    plt.scatter(coords[:, 0],coords[:, 1], c='blue')

    for i in range(len(path) - 1):
        x1, y1 = coords[path[i]]
        x2, y2 = coords[path[i+1]]
        plt.plot([x1,x2], [y1,y2], 'r-', linewidth=1)

    plt.scatter(coords[path[0], 0], coords[path[0], 1], c = 'green', s = 100, label= 'Start')
    plt.scatter(coords[path[-1], 0], coords[path[-1], 1], c='red', s=100, label = 'End')
    plt.title("Best Path So Far")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, format ='jpeg')
    plt.close()
#still need to test/add temperature annealing but basic nearest neighbor is done
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
            #not sure if this is correct way to terminate if path length is too large
            return float('inf'), None
        path_length += best_distance
        path.append(next_node)
        visited.add(next_node)
        
        if path_length >= best_path:
            return float('inf'), None


    path_length += matrix[path[-1]][0]
    path.append(0)

    return path_length, path

def find_best_path():
    #need to provide file path
    file_name = "Dataset_csv/100Triangle300.csv"

    base_name = file_name.rsplit('.', 1)[0]
    coordinate_list = read_coords_as_tuple(file_name)

    dist_matrix = distance_matrix(coordinate_list)

    best_length = float('inf')
    best_path = None
    #waiting for keyboard interrupt
    try:
        while True:
            path_length, new_path = nn_temperature(best_length, dist_matrix)

            if path_length < best_length:
                best_length = path_length
                best_path = new_path
                print(f"New best length is {best_length}")


    #need to have if statement for keyboard interrupt
    except KeyboardInterrupt:
        print("\n Search stopped!")
        print(f"The best path is: {math.ceil(best_length)}")
        plot_path(coordinate_list, best_path, f"{base_name}.jpeg")
        print("\n The path order is: ")
        print(best_path)
        return best_path


if __name__ == "__main__":
    find_best_path()
