import numpy as np
from Losses import mae
from ActivationLayer import initialize_weights, forward, backward, update_weights
from OneHot import encode_labels
import matplotlib.pyplot as plt
import pandas as pd
from Normalize import normalize_data
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


def train_nn(X, y, hidden_size, output_size, epochs, learning_rate):
    input_size = X.shape[1]
    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # Forward Propagation
        hidden_layer_output, output_layer_output = forward(X, weights_input_hidden, weights_hidden_output)

        # Backward Propagation
        hidden_delta, output_delta = backward(X, y, hidden_layer_output, output_layer_output, weights_hidden_output)

        # Update Weights
        weights_input_hidden, weights_hidden_output = update_weights(X, hidden_layer_output, output_delta, hidden_delta,
                                                                     weights_input_hidden, weights_hidden_output,
                                                                     learning_rate)
        # Print loss every 100 epochs
        if epoch % 100 == 0:
            loss = mae(y, output_layer_output)
            print(f"Epoch {epoch}, Training-Loss : {loss:.6f}")

    return weights_input_hidden, weights_hidden_output


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap='BuGn'):
    plt.matshow(df_confusion, cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.title(title)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()

    # Predictions


def predict(X, trained_weights_input_hidden, trained_weights_hidden_output):
    _, y_pred = forward(X, trained_weights_input_hidden, trained_weights_hidden_output)
    y_pred_label = np.argmax(y_pred, axis=1)
    return y_pred_label


def get_confusion_matrix(y_train, y_pred):
    y = pd.Series(y_train, name='Actual')
    y_pred = pd.Series(y_pred, name='Predicted')
    df_confusion = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    return df_confusion


def make_predictions(X, y_true, trained_weights_input_hidden,
                     trained_weights_hidden_output, type):
    # Perform predictions on training data
    y_pred = predict(X, trained_weights_input_hidden, trained_weights_hidden_output)
    # Get Confusion matrix
    df_confusion = get_confusion_matrix(y_true, y_pred)
    print("========================Confusion Matrix========================\n")
    print(df_confusion)
    print()
    print(f'The model {type} accuracy is ={accuracy_score(y_true, y_pred):.5f}\n')
    print(f"The model {type} precision is ={precision_score(y_true, y_pred, average='weighted'):.5f}\n")
    print(f"The model {type} recall is ={recall_score(y_true, y_pred, average='weighted'):.5f}\n")


def perform_pca(X_normalized):
    pca = PCA(n_components=0.9)
    # perform PCA on the scaled data
    pca.fit(X_normalized)
    X_pca = pca.transform(X_normalized)
    return X_pca


if __name__ == '__main__':
    # Load npy data from train and test data.
    logging.info("=======Loading train and train label data========")
    X = np.load('train/X_train.npy')
    y_true = np.load('train/y_train.npy')
    # One-hot encode the labels
    y_one_hot = encode_labels(y_true)
    X_train_normalized = normalize_data(X)

    # Train the neural network
    hidden_size = 16  # Increased hidden size
    output_size = y_one_hot.shape[1]
    epochs = 3000  # Increased number of epochs
    learning_rate = 0.001  # Adjusted learning rate

    logging.info("=======Start training neural network========")
    trained_weights_input_hidden, trained_weights_hidden_output = train_nn(X_train_normalized, y_one_hot,
                                                                           hidden_size, output_size, epochs,
                                                                           learning_rate)
    logging.info("=======Make Predictions on Train Data========")
    make_predictions(X_train_normalized, y_true, trained_weights_input_hidden, trained_weights_hidden_output, 'train')
    print()
    X_test_normalized = normalize_data(np.load('test/X_test.npy'))
    y_test_true = np.load('test/y_test.npy')
    logging.info("=======Make Predictions on Test Data========")
    make_predictions(X_test_normalized, y_test_true, trained_weights_input_hidden,
                     trained_weights_hidden_output, 'test')
    print()
    X_pca_test = perform_pca(X_test_normalized)
    y_one_hot_test = encode_labels(y_test_true)
    logging.info("=======Train Model with PCA Data========")
    trained_weights_input_hidden_pca, trained_weights_hidden_output_pca = train_nn(X_pca_test, y_one_hot_test,
                                                                                   hidden_size, output_size, epochs,
                                                                                   learning_rate)
    print()
    logging.info("=======Make Predictions on PCA Data========")
    make_predictions(X_pca_test, y_test_true, trained_weights_input_hidden_pca,
                     trained_weights_hidden_output_pca, 'pca')
