#!/usr/bin/env python3
import sys

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Wrong number arguments")
        exit(1)


    # writing to file
    file1 = open(str(sys.argv[1]), 'r')
    file2 = open(str(sys.argv[2]), 'r')
    Lines1 = file1.readlines()
    Lines2 = file2.readlines()

    if len(Lines1) != len(Lines2):
        print("unequal items")
        exit(1)
    
    for i in range(len(Lines1)):
        data1 = float(Lines1[i])
        data2 = float(Lines2[i])

        if data1 * data2 < 0:
            print(f"not equal sign at index {i}, data1 = {data1}, data2 = {data2}")

        error = (data1 - data2) / max(data1, data2)

        if(error > 0.01):
            print(f"mismatch at index {i}, data1 = {data1}, data2 = {data2}") 