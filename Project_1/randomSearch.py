import sys
import math

def calculateDistance(xOne, xTwo, yOne, yTwo):
    return math.sqrt(math.pow((xTwo - xOne), 2) - math.pow((yTwo - yOne), 2))



fileName = str(input("Enter the name of file: "))

coords = []
with open(fileName, 'r') as file:
    for x in file:
        pairOfCoords = x.strip().split()
        x = float(pairOfCoords[0])
        y = float(pairOfCoords[1])
        coords.append((x,y))