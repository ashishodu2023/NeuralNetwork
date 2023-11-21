import numpy as np
from FlattenLayer import FlattenLayer
from FullyConnectedLayer import FCLayer
from ActivationLayer import ActivationLayer
from SoftmaxLayer import SoftmaxLayer
from Activations import relu, relu_prime
from Losses import  mse, mse_prime,mae,mae_prime
import matplotlib.pyplot as plt

def main():
    X_train = np.load('train/X_train.npy')
    y_train = np.load('train/y_train.npy')

    network = [
        FCLayer(10, 10),
        ActivationLayer(relu, relu_prime),
        FCLayer(10, 8),
        ActivationLayer(relu, relu_prime),
        FCLayer(8, 10),
        ActivationLayer(relu, relu_prime),
        SoftmaxLayer(10)
    ]

    epochs = 20
    learning_rate = 0.01

    # training
    for epoch in range(epochs):
        loss = 0.0
        for x, y_true in zip(X_train, y_train):
            # forward
            output = x
            for layer in network:
                output = layer.forward(output)

            # error (display purpose only)
            loss += mae(y_true, output)

            # backward
            output_error = mae_prime(y_true, output)
            for layer in reversed(network):
                output_error = layer.backward(output_error, learning_rate)

        loss /= len(X_train)
        print('%d/%d, Loss=%f' % (epoch + 1, epochs, loss))
    print()


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
    error = sum([mae(y, predict(network, x)) for x, y in zip(X_train, y_train)]) / len(X_train)
    print('Ratio: %.2f' % ratio)
    print('MAE: %.4f' % error)
    print()

    yhat = predict(network, np.load('test/X_test.npy'))
    print(yhat[0])
    print(y_train[0])

    """
    y = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
    y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')
    df_confusion = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(df_confusion)

    # df_confusion = pd.crosstab(y, y_pred)
    plot_confusion_matrix(df_confusion)
    """


if __name__=='__main__':
    main()