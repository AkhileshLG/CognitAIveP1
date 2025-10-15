import csv
from typing import List, Tuple
import numpy as np

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


#still need to test/add temperature annealing but basic nearest neighbor is done
def nn_temperature(matrix):

    path = [0]
    visited = {0}
    path_length = 0

    for i in range(len(matrix) - 1):
        curr_node = path[-1]
        next_node = None
        best_distance = float('inf')
        
        for j in range(len(matrix)):
            if curr_node in visited:
                continue
            curr_distance = matrix[curr_node][j]

            if curr_distance < best_distance:
                next_node = curr_node
                best_distance = curr_distance
                
        path_length += best_distance
        path.append(next_node)
        visited.add(next_node)

    path_length += matrix[path[-1]][0]
    path.append(0)

    return path_length

def find_best_path():
    #need to provide file path
    coordinate_list = read_coords_as_tuple()

    dist_matrix = distance_matrix(coordinate_list)

    best_path = float('inf')
    #waiting for keyboard interrupt
    while True:
        new_path = nn_temperature(dist_matrix)

        if new_path < best_path:
            best_path = new_path

    return best_path





    
    
    