import numpy as np
from FlattenLayer import FlattenLayer
from FullyConnectedLayer import FCLayer
from ActivationLayer import ActivationLayer
from SoftmaxLayer import SoftmaxLayer
from Activations import relu, relu_prime
from Losses import sse, sse_prime, mse, mse_prime
import pandas as pd
import matplotlib.pyplot as plt

def main():
    X_train = np.load('data/feature.npy')
    y_train = np.load('data/label.npy')

    network = [
        FlattenLayer(input_shape=(1024, 1)),
        FCLayer(1024, 1024),
        ActivationLayer(relu, relu_prime),
        FCLayer(1024, 10),
        SoftmaxLayer(1)
    ]

    epochs = 20
    learning_rate = 0.001

    # training
    for epoch in range(epochs):
        loss = 0.0
        for x, y_true in zip(X_train, y_train):
            # forward
            output = x
            for layer in network:
                output = layer.forward(output)

            # error (display purpose only)
            loss += mse(y_true, output)

            # backward
            output_error = mse_prime(y_true, output)
            for layer in reversed(network):
                output_error = layer.backward(output_error, learning_rate)

        loss /= len(X_train)
        print('%d/%d, Loss=%f' % (epoch + 1, epochs, loss))


    # Confusion Matrix
    def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
        plt.matshow(df_confusion, cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(df_confusion.columns))
        plt.xticks(tick_marks, df_confusion.columns, rotation=45)
        plt.yticks(tick_marks, df_confusion.index)
        plt.ylabel(df_confusion.index.name)
        plt.xlabel(df_confusion.columns.name)
        plt.show()


    # Predictions
    def predict(network, input):
        output = input
        for layer in network:
            output = layer.forward(output)
        return output


    ratio = sum([np.argmax(y) == np.argmax(predict(network, x)) for x, y in zip(X_train, y_train)]) / len(X_train)
    error = sum([mse(y, predict(network, x)) for x, y in zip(X_train, y_train)]) / len(X_train)
    print('ratio: %.2f' % ratio)
    print('mse: %.4f' % error)

    y = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
    y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')
    df_confusion = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

    # df_confusion = pd.crosstab(y, y_pred)
    plot_confusion_matrix(df_confusion)

if __name__=='__main__':
    main()