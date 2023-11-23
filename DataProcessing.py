import numpy as np
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

with open('file.txt','r') as file_obj:
    logging.info("====Open the file for reading====")
    binary_sequences_with_labels = file_obj.readlines()
# Split the input into sequences and labels
binary_sequences_with_labels_dict = dict()
binary_sequences_with_labels_list =[]
i = 0
# iterating until the end of the file
logging.info("====Starting while loop====")
while i < len(binary_sequences_with_labels):
    # extracting 32 lines (image matrix)
    data_lines_list = []
    for j in range(i, i + 32):
        # reading each line as string from i to i+32 to get the image matrix lines
        single_line = binary_sequences_with_labels[j].strip()
        # add these string lines into a temp list
        data_lines_list.append(list(map(int, single_line)))  # converting each character to int

    # extracting the label from the 33rd line
    label = int(binary_sequences_with_labels[j + 1].strip())

    # concatenate the list of lines to form a matrix
    data_matrix = np.array(data_lines_list).flatten()

    # append label and data matrix to the list
    binary_sequences_with_labels_list.append([label, data_matrix])

    # move to the next block of data for the image matrix and its label
    i += 33
logging.info("====Get data and labels====")
labels, data_matrix = zip(*binary_sequences_with_labels_list)
X = np.array(data_matrix)
y= np.array(labels)

logging.info("====Saving complete data into data directory====")
np.save('data/X.npy', X)
np.save('data/y.npy', y)
print('The shape of X and y is',X.shape,y.shape)

logging.info("====Split the data into train and test sets====")
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
print('The shape of Training and Test dataset are', X_train.shape,X_test.shape,y_train.shape,y_test.shape)

logging.info("====Saving data into train and test npy files====")
np.save('train/X_train.npy', X_train)
np.save('test/X_test.npy', X_test)

# Save labels as well
np.save('train/y_train.npy', np.array(y_train))
np.save('test/y_test.npy', np.array(y_test))

