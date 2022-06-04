import numpy as np
from sklearn import linear_model


def compile_data(full_data, months_in_feature=12, months_ahead_predict=12, num_cities=17):
    with open(full_data) as f:
        ncols = len(f.readline().split(','))
    other_features = ncols - num_cities - 1
    full = np.loadtxt(full_data, delimiter=',', skiprows=1, usecols=[*range(0, num_cities), *range(num_cities+1, ncols)])
    data = np.empty((1, num_cities + (other_features * months_in_feature)))
    labels = np.empty(0)
    for i in range(num_cities):
        mask = (full[:, i] == float(1))
        cur = full[mask, :]
        one_hot = cur[0, :num_cities]
        for start in range(cur.shape[0] - (months_in_feature + months_ahead_predict)):
            newrow = one_hot
            for idx in range(start, start + months_in_feature):
                newrow = np.append(newrow, cur[idx, num_cities:], axis=0)
            data = np.vstack([data, newrow])
            #case shiller index to try to predict
            val = cur[start + months_in_feature + months_ahead_predict, -6]
            labels = np.append(labels, val)
    data = np.delete(data, 0, axis=0)
    labels.shape = (len(labels),)

    # Generate random selection of rows to split dataset
    num_train = int(len(labels))
    mask = np.random.choice([True, False], len(labels), p=[0.85, 0.15])
    anti_mask = ~mask

    train_data = data[mask]
    train_labels = labels[mask]
    val_data = data[anti_mask]
    val_labels = labels[anti_mask]

    return train_data, train_labels, val_data, val_labels


def linreg(train_data, train_labels, val_data, val_labels):
    # linear regression model
    lin = linear_model.LinearRegression()
    lin.fit(train_data, train_labels)

    pred = lin.predict(val_data)
    percent_err = np.abs(pred - val_labels) / val_labels

    return percent_err


def main():
    train_data, train_labels, val_data, val_labels = compile_data("FINAL_all_data_one_hot.csv")
    err1 = linreg(train_data, train_labels, val_data, val_labels)
    print(np.mean(err1))


if __name__ == '__main__':
    main()
