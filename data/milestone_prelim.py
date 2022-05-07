import numpy as np
import csv

def load_data(full_data):
    with open(full_data, 'r', newline='') as csv_fh:
        headers = csv_fh.readline().strip().split(',')
    
    data = np.loadtxt(full_data, delimiter = ',', skiprows = 1)

    print(data[0])

def main():
    load_data("all_data_w_city_names.csv")

if __name__ == '__main__':
    main()