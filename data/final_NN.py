import tensorflow as tf
import numpy as np

CITIES = ['atlanta', 'boston', 'chicago', 'cleveland', 'dallas', 'denver', 'detroit', 'la', 'miami',
          'minneapolis', 'nyc', 'phoenix', 'portland', 'sf', 'seattle', 'tampa', 'dc']


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

def NN(data, labels):
    mnist = tf.keras.datasets.mnist

    data = np.loadtxt(data, delimiter=',')
    labels = np.loadtxt(labels, delimiter=',')

    model_1relu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = data.shape[1]),
             tf.keras.layers.Dense(128, activation = 'relu'), #tf.keras.layers.Dense(n_units, activation = tf.keras.layers.LeakyReLU(alpha = 0.01))
             tf.keras.layers.Dense(1)
             ])

    model_3relu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = data.shape[1]),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(1)
             ])

    model_5relu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = data.shape[1]),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(1)
             ])

    model_10relu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = data.shape[1]),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(1)
             ])

    model_20relu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = data.shape[1]),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(128, activation = 'relu'),
             tf.keras.layers.Dense(1)
             ])

    model_1Lrelu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = data.shape[1]),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(1)
             ])

    model_3Lrelu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = data.shape[1]),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(1)
             ])

    model_5Lrelu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = data.shape[1]),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(1)
             ])

    model_10Lrelu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = data.shape[1]),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(1)
             ])

    model_20Lrelu = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(input_shape = data.shape[1]),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(128, activation = tf.keras.layers.LeakyReLU(alpha = 0.01)),
             tf.keras.layers.Dense(1)
             ])
    
    loss_fn = tf.keras.losses.MeanSquaredError()

    model_1relu.compile(optimizer = 'adam', loss = loss_fn, metrics = [tf.keras.metrics.MeanAbsolutePercentageError()])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5,
                                                  restore_best_weights = True)
    model_1relu.fit(x_train, y_train, batch_size = 32, epochs = 20, validation_data = (x_val, y_val),
                    callbacks = [early_stop])

    model_1relu.evaluate(x_test,  y_test, verbose = 2)

def main():
    print("Hello")

if __name__ == "__main__":
    print(num_features("one_hot_12feature_12predict.csv"))
    #NN("one_hot_12feature_12predict.csv", "labels_12feature_12predict.csv")