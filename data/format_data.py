import numpy as np


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
            val = cur[start + months_in_feature + months_ahead_predict - 1, -6]
            labels = np.append(labels, val)
    data = np.delete(data, 0, axis=0)
    labels.shape = (len(labels),)
    np.savetxt("one_hot_" + str(months_in_feature) + "feature_" + str(months_ahead_predict) + "predict.csv", data,
               delimiter=',')
    np.savetxt("labels_" + str(months_in_feature) + "feature_" + str(months_ahead_predict) + "predict.csv", labels,
               delimiter=',')


if __name__ == "__main__":
    compile_data("FINAL_all_data_one_hot.csv", months_in_feature=18, months_ahead_predict=12)
