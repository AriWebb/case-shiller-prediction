import numpy as np
import csv
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import svm

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
    #linear regression model
    lin = linear_model.LinearRegression()
    lin.fit(train_data, train_labels)
    
    pred = lin.predict(val_data)
    percent_err = np.abs(pred - val_labels) / val_labels

    return percent_err

def test2(train_data, train_labels, val_data, val_labels):
    # performs basic f_regression feature selection to see which features are most correlated with case-schilling
    feature_names = "atlanta,boston,chicago,cleveland," \
                    "dallas,denver,detroit,la,miami,minneapolis,nyc," \
                    "phoenix,portland,sf,seattle,tampa,dc,cpi,crimes_reported," \
                    "crimes_cleared,patents,population,unemployment,case_shiller,dow,nasdaq,sp"
    feature_names = feature_names.split(',')
    selector = SelectKBest(f_regression, k=10)
    selector.fit(train_data, train_labels)
    mask = selector.get_support()
    best_features = []
    for b, feature in zip(mask, feature_names):
        if b:
            best_features.append(feature)
    return best_features

def test3(train_data, train_labels, val_data, val_labels):
    #svm model
    regr = svm.SVR()
    regr.fit(train_data, train_labels)
    preds = regr.predict(val_data)
    percent_err = np.abs(preds - val_labels) / val_labels
    return percent_err

def main():
    train_data, train_labels, val_data, val_labels = load_data("all_data_w_city_names_no_date.csv")
    err1 = test1(train_data, train_labels, val_data, val_labels)
    print(np.mean(err1))
    print(test2(train_data, train_labels, val_data, val_labels))
    err2 = test3(train_data, train_labels, val_data, val_labels)
    print(np.mean(err2))

if __name__ == '__main__':
    main()