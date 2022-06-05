import tensorflow as tf
import numpy as np

CITIES = ['atlanta', 'boston', 'chicago', 'cleveland', 'dallas', 'denver', 'detroit', 'la', 'miami',
          'minneapolis', 'nyc', 'phoenix', 'portland', 'sf', 'seattle', 'tampa', 'dc']

def num_features(data):
    data = np.loadtxt(data, delimiter=',')
    result = {}

    for i in range(len(CITIES)):
        result[CITIES[i]] = int(np.sum(data[:, i]))

    return result

def NN(data, labels):
    mnist = tf.keras.datasets.mnist

    data = np.loadtxt(data, delimiter=',')
    labels = np.loadtxt(labels, delimiter=',')

    model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape = data.shape[1]),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
    ])

    loss_fn = 0

    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test,  y_test, verbose=2)

def main():
    print("Hello")

if __name__ == "__main__":
    print(num_features("one_hot_12feature_12predict.csv"))
    #NN("one_hot_12feature_12predict.csv", "labels_12feature_12predict.csv")