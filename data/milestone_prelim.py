import numpy as np
import csv
from sklearn import linear_model

def load_data(full_data):
    with open(full_data, 'r', newline='') as csv_fh:
        headers = csv_fh.readline().strip().split(',')
    
    # Loads all data into numpy array
    data = np.loadtxt(full_data, delimiter = ',', skiprows = 1)

    all_features = data[:, :-1]
    all_labels = data[:, -1]

    # Generate random selection of rows to split dataset
    num_train = int(len(all_labels))
    mask = np.random.choice([True, False], len(all_labels), p = [0.85, 0.15])
    anti_mask = ~mask
    
    train_data = all_features[mask]
    train_labels = all_labels[mask]
    val_data = all_features[anti_mask]
    val_labels = all_labels[anti_mask]

    return train_data, train_labels, val_data, val_labels

def test1(train_data, train_labels, val_data, val_labels):
    lin = linear_model.LinearRegression()
    lin.fit(train_data, train_labels)
    
    pred = lin.predict(val_data)
    percent_err = np.abs(pred - val_labels) / val_labels

    return percent_err

def test2(train_data, train_labels, val_data, val_labels):
    # feature transformation
    return 0

def test3(train_data, train_labels, val_data, val_labels):
    # less braindead model
    return 0

def main():
    train_data, train_labels, val_data, val_labels = load_data("all_data_w_city_names_no_date.csv")
    err1 = test1(train_data, train_labels, val_data, val_labels)
    print(np.mean(err1))

if __name__ == '__main__':
    main()