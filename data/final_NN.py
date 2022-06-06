import tensorflow as tf
import numpy as np
import scipy.stats as sts
import pandas as pd
import json

CITIES = ['atlanta', 'boston', 'chicago', 'cleveland', 'dallas', 'denver', 'detroit', 'la', 'miami',
          'minneapolis', 'nyc', 'phoenix', 'portland', 'sf', 'seattle', 'tampa', 'dc']

LABEL = ["cpi", "violent crime", "property crime", "patents", "population", "unemployment", "case shiller",\
          "education and health", "medical care", "motor fuel", "income", "food and bev"]

def test():
    data = np.loadtxt("one_hot_12feature_12predict.csv", delimiter=',')
    labels = np.loadtxt("labels_12feature_12predict.csv", delimiter=',')
    train_feat, train_labels, val_feat, val_labels, test_feat, test_labels = \
        gen_data("one_hot_12feature_12predict.csv", "labels_12feature_12predict.csv")

    # (1.088888888888889, 5.373319498591078)

    cs_rec = val_feat[:, -6]

    label_names = CITIES
    for i in range(1, 13):
        for lab in LABEL:
            add = lab + "_" + str(i).zfill(2)
            label_names.append(add)

    df = pd.DataFrame(np.c_[train_feat, train_labels.reshape(-1, 1)], columns = label_names + ["Prediction"])
    #print(df.corr(method = "pearson")["Prediction"][:].sort_values(ascending = False, key=abs).head(20), "\n\n\n")
    #print(df.corr(method = "spearman")["Prediction"][:].sort_values(ascending = False, key=abs).head(50))
    #print(df.corr(method = "kendall")["Prediction"][:].sort_values(ascending = False, key=abs).head(50))

    with open('NN_top1.json') as json_file:
        NN_data = json.load(json_file)

    NN_list = list()
    for key in NN_data.keys():
        NN_list.append([key, NN_data[key]])
    
    MAPE = np.array([row[1] for row in NN_list])
    ind = np.argpartition(MAPE, 20)[:20]
    ind = ind[np.argsort(MAPE[ind])]

    for i in ind:
        print(NN_list[i])


def num_features(data):
    data = np.loadtxt(data, delimiter=',')
    result = {}
    indices = {}

    total = 0
    for i in range(len(CITIES)):
        result[CITIES[i]] = int(np.sum(data[:, i]))
        indices[CITIES[i]] = (total, total + int(np.sum(data[:, i])) - 1)
        total += int(np.sum(data[:, i]))

    return result, indices

def gen_data(data, labels, silly = False):
    dataset = np.loadtxt(data, delimiter=',')
    labels = np.loadtxt(labels, delimiter=',')
    examples, indices = num_features(data)
    train_feat = list()
    train_labels = list()
    val_feat = list()
    val_labels = list()
    test_feat = list()
    test_labels = list()

    for city in CITIES:
        val = int(0.9 * examples[city])
        train = int(0.75 * examples[city])

        train_feat += list(dataset[indices[city][0] : indices[city][0] + train])
        train_labels += list(labels[indices[city][0] : indices[city][0] + train])
        val_feat += list(dataset[indices[city][0] + train : indices[city][0] + val])
        val_labels += list(labels[indices[city][0] + train : indices[city][0] + val])
        test_feat += list(dataset[indices[city][0] + val : indices[city][1] + 1])
        test_labels += list(labels[indices[city][0] + val: indices[city][1] + 1])

    if silly:
        food_ind = [-1]
        inc_ind = [-2]
        motor_ind = [-3]
        med_ind = [-4]
        edu_ind = [-5]
        cs_ind = [-6]
        unemp_ind = [-7]
        pop_ind = [-8]
        pat_ind = [-9]
        prop_ind = [-10]
        vio_ind = [-11]
        cpi_ind = [-12]

        city_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        for i in range(11):
            food_ind.append(food_ind[-1] - 12)
            inc_ind.append(inc_ind[-1] - 12)
            motor_ind.append(motor_ind[-1] - 12)
            med_ind.append(med_ind[-1] - 12)
            edu_ind.append(edu_ind[-1] - 12)
            cs_ind.append(cs_ind[-1] - 12)
            unemp_ind.append(unemp_ind[-1] - 12)
            pop_ind.append(pop_ind[-1] - 12)
            pat_ind.append(pat_ind[-1] - 12)
            prop_ind.append(prop_ind[-1] - 12)
            vio_ind.append(vio_ind[-1] - 12)
            cpi_ind.append(cpi_ind[-1] - 12)

        for i in range(len(train_feat)):
            train_feat[i] = train_feat[i][cs_ind] #np.concatenate((train_feat[i][inc_ind], train_feat[i][cs_ind]))

        for i in range(len(val_feat)):
            val_feat[i] = val_feat[i][cs_ind] #np.concatenate((val_feat[i][inc_ind], val_feat[i][cs_ind]))

        for i in range(len(test_feat)):
            test_feat[i] = test_feat[i][cs_ind] #np.concatenate((test_feat[i][cpi_ind], test_feat[i][cs_ind]))

        print(len(train_feat[0]))

    train_feat = np.array(train_feat)
    train_labels = np.array(train_labels)
    val_feat = np.array(val_feat)
    val_labels = np.array(val_labels)
    test_feat = np.array(test_feat)
    test_labels = np.array(test_labels)

    return train_feat, train_labels, val_feat, val_labels, test_feat, test_labels


def relu_layer_list(num_hidden_layers, num_neurons, input_shape, act):
    layers = [tf.keras.layers.Flatten(input_shape = (input_shape,))]
    for i in range(num_hidden_layers):
        layers.append(tf.keras.layers.Dense(num_neurons, activation = act))
    layers.append(tf.keras.layers.Dense(1))
    return layers

def NN_tests(data, labels):
    train_feat, train_labels, val_feat, val_labels, test_feat, test_labels = gen_data(data, labels, True)
    
    adam = tf.keras.optimizers.Adam()
    nadam = tf.keras.optimizers.Nadam()
    adamax = tf.keras.optimizers.Adamax()
    adadelta = tf.keras.optimizers.Adadelta() # More robust Adagrad
    sgd = tf.keras.optimizers.SGD()
    rms = tf.keras.optimizers.RMSprop()

    optimizers = [(adam, "Adam"), (nadam, "Nadam"), (adamax, "Adamax"), (adadelta, "Adadelta"), (sgd, "SGD"), (rms, "RMSprop")]
    batch_sizes = [16, 32, 64]
    layers = [1, 2, 5, 10]
    neurons = [3, 6, 12, 24]
    activations = [("relu", "relu"), ("Lrelu", tf.keras.layers.LeakyReLU())]

    evaluation = {}

    loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10,
                                                  restore_best_weights=True)

    # Hyperparameter Optimization

    for opt, opt_name in optimizers:
        for batch in batch_sizes:
            for layer in layers:
                for neu in neurons:
                    for act_name, act in activations:
                        model_name = "O = " + opt_name + "; B = " + str(batch) + "; L = " + str(layer) + "; N = " + str(neu) + \
                                     "; A = " + act_name
                        print("Starting model: ", model_name)

                        model = tf.keras.models.Sequential(relu_layer_list(layer, neu, train_feat.shape[1], act))

                        model.compile(optimizer = opt, loss = loss_fn, metrics = [tf.keras.metrics.MeanSquaredError()])
                        model.fit(train_feat, train_labels, batch_size = batch, epochs = 50, validation_data = (val_feat, val_labels),
                                callbacks = [early_stop], verbose = 0, steps_per_epoch = train_feat.shape[0] // (10 * batch), shuffle = True)

                        evaluation[model_name] = model.evaluate(val_feat, val_labels, verbose = 2)

    print(evaluation)

    with open('NN.json', 'w') as fp:
        json.dump(evaluation, fp, indent = 6)

def NN():
    train_feat, train_labels, val_feat, val_labels, test_feat, test_labels = gen_data("one_hot_12feature_12predict.csv",\
                                                                                      "labels_12feature_12predict.csv", True)

    loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20,
                                                  restore_best_weights=True)

    model = tf.keras.models.Sequential(relu_layer_list(3, 24, train_feat.shape[1], tf.keras.layers.LeakyReLU()))

    model.compile(optimizer = "adam", loss = loss_fn, metrics = [tf.keras.metrics.MeanSquaredError()])
    
    for i in range(10):
        print(i)
        model.fit(train_feat, train_labels, batch_size = 16, epochs = 100, validation_data = (val_feat, val_labels),
              callbacks = [early_stop], verbose = 0, shuffle = True)

    model.evaluate(test_feat, test_labels, verbose = 2)

def main():
    print("Hello")

if __name__ == "__main__":
    #test()
    #gen_data("one_hot_12feature_12predict.csv", "labels_12feature_12predict.csv")
    NN()