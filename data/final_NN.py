import tensorflow as tf
import numpy as np

CITIES = ['atlanta', 'boston', 'chicago', 'cleveland', 'dallas', 'denver', 'detroit', 'la', 'miami',
          'minneapolis', 'nyc', 'phoenix', 'portland', 'sf', 'seattle', 'tampa', 'dc']

def test():
    data = np.loadtxt("labels_12feature_12predict.csv", delimiter=',')
    print(data[0])

def weighted_mean_squared(y_actual, y_pred):
    """
    A mean squared error loss function that penalizes over-predictions more than
    under under-predictions.
    """
    if y_actual < y_pred:
        loss = tf.keras.backend.square((y_actual - y_pred) * 1.1)
    else:
        loss = tf.keras.backend.square(y_actual - y_pred)
    return loss

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

def gen_data(data, labels):
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

    train_feat = np.array(train_feat)
    train_labels = np.array(train_labels)
    val_feat = np.array(val_feat)
    val_labels = np.array(val_labels)
    test_feat = np.array(test_feat)
    test_labels = np.array(test_labels)

    return train_feat, train_labels, val_feat, val_labels, test_feat, test_labels

def NN(data, labels):
    mnist = tf.keras.datasets.mnist

    train_feat, train_labels, val_feat, val_labels, test_feat, test_labels = gen_data(data, labels)

    model_1relu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = (train_feat.shape[1],)),
             tf.keras.layers.Dense(80, activation = 'relu'), #tf.keras.layers.Dense(n_units, activation = tf.keras.layers.LeakyReLU(alpha = 0.01))
             tf.keras.layers.Dense(1)
             ])

    model_3relu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = (train_feat.shape[1],)),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(1)
             ])

    model_5relu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = (train_feat.shape[1],)),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(1)
             ])

    model_10relu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = (train_feat.shape[1],)),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(1)
             ])

    model_20relu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = (train_feat.shape[1],)),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(80, activation = 'relu'),
             tf.keras.layers.Dense(1)
             ])

    # tf.keras.layers.Dense(80, activation = tf.keras.layers.LeakyReLU(alpha = 0.01))

    models = [("model_1relu", model_1relu), ("model_3relu", model_3relu), ("model_5relu", model_5relu),
              ("model_10relu", model_10relu), ("model_20relu", model_20relu),]
    loss_fn = tf.keras.losses.MeanSquaredError()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5,
                                                  restore_best_weights=True)

    for model_name, model in models:
        print('============================================\n' + model_name)
        model.compile(optimizer = 'adam', loss = loss_fn, metrics = [tf.keras.metrics.MeanAbsolutePercentageError()])
        model.fit(train_feat, train_labels, batch_size = 32, epochs = 20, validation_data = (val_feat, val_labels),
                        callbacks = [early_stop])
        model.evaluate(test_feat, test_labels, verbose = 2)
        print('============================================')

def main():
    print("Hello")

if __name__ == "__main__":
    #gen_data("one_hot_12feature_12predict.csv", "labels_12feature_12predict.csv")
    NN("one_hot_12feature_12predict.csv", "labels_12feature_12predict.csv")