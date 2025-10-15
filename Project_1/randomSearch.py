import sys

def calculateDistance(xOne, xTwo, yOne, yTwo):
    return (xTwo - xOne) + (yTwo - yOne)

def main():
    fileName = str(input("Enter the name of file: "))
    yesOrNo = (input("Run Program (y/n): "))

    if yesOrNo == 'y':
        print()
    elif yesOrNo == 'n':
        sys.exit()