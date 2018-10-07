import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer


def read_data_and_labels(data_location, labels_location):
    with open(data_location) as data_set, open(labels_location) as labels:
        data = data_set.readlines()
        labels = labels.readlines()
        print('Found {} reviews and {} labels'.format(len(data), len(labels)))

        return data, labels


def get_data_as_one_hot(num_words, data_location='data/data', labels_location='data/labels'):
    data, labels = read_data_and_labels(data_location, labels_location)

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(data)
    one_hot = tokenizer.texts_to_matrix(data, mode='binary')
    encoded_labels = np.asarray(labels).astype('float32')

    print('Returning encoded text, labels and tokenizer')
    return one_hot, encoded_labels, tokenizer


def get_data_as_padded_sequences(num_words, max_length, data_location='data/data', labels_location='data/labels'):
    data, labels = read_data_and_labels(data_location, labels_location)

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    sequences = pad_sequences(sequences, maxlen=max_length)

    encoded_labels = np.asarray(labels).astype('float32')

    return sequences, encoded_labels, tokenizer
