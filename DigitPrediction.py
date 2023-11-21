import numpy as np
import pandas as pd
from FullyConnectedLayer import FCLayer
from ActivationLayer import ActivationLayer
from SoftmaxLayer import SoftmaxLayer
from Activations import relu, relu_prime
from Losses import sse, sse_prime
import matplotlib.pyplot as plt
from itertools import chain


def model_configuration():
    # Configure model with layers and activation function.
    model = [
        FCLayer(10, 8),
        ActivationLayer(relu, relu_prime),
        FCLayer(8, 4),
        ActivationLayer(relu, relu_prime),
        FCLayer(4, 1),
        ActivationLayer(relu, relu_prime),
        SoftmaxLayer(1)
    ]
    return model

def training(model,X_train,y_train,learning_rate):
    # Training
    for epoch in range(epochs):
        loss = 0.0
        for x, y_true in zip(X_train, y_train):
            # forward
            output = x
            for layer in model:
                output = layer.forward(output)

            # error (display purpose only)
            loss += sse(y_true, output)

            # backward
            output_error = sse_prime(y_true, output)
            for layer in reversed(model):
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
def predict(model, input):
    output = input
    for layer in model:
        output = layer.forward(output)
    return output

def get_ratio_error(model,X_train,y_train):
    ratio = sum([np.argmax(y) == np.argmax(predict(model, x)) for x, y in zip(X_train, y_train)]) / len(X_train)
    error = sum([sse(y, predict(model, x)) for x, y in zip(X_train, y_train)]) / len(X_train)
    print('Ratio: %.2f' % ratio)
    print('SSE: %.4f' % error)
    print()

def get_confusion_matrix(model,y_train,y_pred):
    y = pd.Series(y_train, name='Actual')
    y_pred = pd.Series(y_pred, name='Predicted')
    df_confusion = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    return df_confusion



if __name__ == '__main__':
    # Load npy data from train and test data.
    X_train = np.load('train/X_train.npy')
    y_train = np.load('train/y_train.npy')

    # Hyperparameters
    epochs = 5
    learning_rate = 0.001

    # Get model configurations
    model=model_configuration()

    # Start Model Training
    training(model,X_train, y_train, learning_rate)

    # Get predictions
    y_pred = list(chain.from_iterable(predict(model, np.load('test/X_test.npy'))))

    # Get error ratio
    get_ratio_error(model, X_train, y_train)

    # Get Confusion matrix
    df_confusion=get_confusion_matrix(model, y_train,y_pred)
    print(df_confusion)

    # Plot confusion matrix
    #plot_confusion_matrix(df_confusion)
