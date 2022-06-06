import numpy as np
from sklearn import linear_model
from final_NN import gen_data


def linreg(train_data, train_labels, val_data, val_labels):
    # linear regression model
    lin = linear_model.LinearRegression()
    lin.fit(train_data, train_labels)

    pred = lin.predict(val_data)
    percent_err = np.abs(pred - val_labels) / val_labels

    return percent_err


def main():
    data3 = "one_hot_3feature_12predict.csv"
    labels3 = "labels_3feature_12predict.csv"
    train_feat3, train_labels3, val_feat3, val_labels3, test_feat3, test_labels3 = gen_data(data3, labels3)
    err3 = linreg(train_feat3, train_labels3, val_feat3, val_labels3)
    print("3 months in the features predicting 12 months in the future: " + str(np.mean(err3)))

    data6 = "one_hot_6feature_12predict.csv"
    labels6 = "labels_6feature_12predict.csv"
    train_feat6, train_labels6, val_feat6, val_labels6, test_feat6, test_labels6 = gen_data(data6, labels6)
    err6 = linreg(train_feat6, train_labels6, val_feat6, val_labels6)
    print("6 months in the features predicting 12 months in the future: " + str(np.mean(err6)))

    data12 = "one_hot_12feature_12predict.csv"
    labels12 = "labels_12feature_12predict.csv"
    train_feat12, train_labels12, val_feat12, val_labels12, test_feat12, test_labels12 = gen_data(data12, labels12)
    err12 = linreg(train_feat12, train_labels12, val_feat12, val_labels12)
    print("12 months in the features predicting 12 months in the future: " + str(np.mean(err12)))
    test_err12 = linreg(train_feat12, train_labels12, test_feat12, test_labels12)
    print("TEST 12 months in the features predicting 12 months in the future: " + str(np.mean(test_err12)))

    data18 = "one_hot_18feature_12predict.csv"
    labels18 = "labels_18feature_12predict.csv"
    train_feat18, train_labels18, val_feat18, val_labels18, test_feat18, test_labels18 = gen_data(data18, labels18)
    err18 = linreg(train_feat18, train_labels18, val_feat18, val_labels18)
    print("18 months in the features predicting 12 months in the future: " + str(np.mean(err18)))

    data24 = "one_hot_24feature_12predict.csv"
    labels24 = "labels_24feature_12predict.csv"
    train_feat24, train_labels24, val_feat24, val_labels24, test_feat24, test_labels24 = gen_data(data24, labels24)
    err24 = linreg(train_feat24, train_labels24, val_feat24, val_labels24)
    print("24 months in the features predicting 12 months in the future: " + str(np.mean(err24)))

    data36 = "one_hot_36feature_12predict.csv"
    labels36 = "labels_36feature_12predict.csv"
    train_feat36, train_labels36, val_feat36, val_labels36, test_feat36, test_labels36 = gen_data(data36, labels36)
    err36 = linreg(train_feat36, train_labels36, val_feat36, val_labels36)
    print("36 months in the features predicting 12 months in the future: " + str(np.mean(err36)))


if __name__ == '__main__':
    main()
