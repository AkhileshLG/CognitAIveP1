import csv
import numpy as np
from typing import Tuple, List, Union

def read_csv_coordinates(file_path: str) -> Tuple[List[float], List[float]]:
    x_coords = []
    y_coords = []
    
    try:
        with open(file_path, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            
            for row_num, row in enumerate(csv_reader, 1):
                if len(row) != 2:
                    raise ValueError(f"Row {row_num} must contain exactly 2 values (x, y), found {len(row)}")
                
                try:
                    x = float(row[0])
                    y = float(row[1])
                    x_coords.append(x)
                    y_coords.append(y)
                except ValueError as e:
                    raise ValueError(f"Row {row_num}: Unable to convert values to float - {e}")
                    
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    return x_coords, y_coords


def read_csv_coordinates_as_array(file_path: str) -> np.ndarray:
    coordinates = []
    
    try:
        with open(file_path, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            
            for row_num, row in enumerate(csv_reader, 1):
                if len(row) != 2:
                    raise ValueError(f"Row {row_num} must contain exactly 2 values (x, y), found {len(row)}")
                
                try:
                    x = float(row[0])
                    y = float(row[1])
                    coordinates.append([x, y])
                except ValueError as e:
                    raise ValueError(f"Row {row_num}: Unable to convert values to float - {e}")
                    
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    return np.array(coordinates)


def read_csv_coordinates_pandas(file_path: str) -> Tuple[List[float], List[float]]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for this function. Install with: pip install pandas")
    
    try:
        df = pd.read_csv(file_path, header=None, names=['x', 'y'])
        return df['x'].tolist(), df['y'].tolist()
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")


# Example usage and demonstration
if __name__ == "__main__":
    # Example usage with one of the existing CSV files
    csv_file = "Project_1/Dataset_csv/128Circle201.csv"
    
    try:
        # Method 1: Using lists
        x_coords, y_coords = read_csv_coordinates(csv_file)
        print(f"Read {len(x_coords)} coordinate pairs")
        print(f"First 5 coordinates:")
        for i in range(min(5, len(x_coords))):
            print(f"  ({x_coords[i]:.2f}, {y_coords[i]:.2f})")
        
        # Method 2: Using numpy array
        coords_array = read_csv_coordinates_as_array(csv_file)
        print(f"\nNumpy array shape: {coords_array.shape}")
        print(f"First 3 rows:\n{coords_array[:3]}")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
