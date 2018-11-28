import random
import sys
import numpy as np
from keras import layers, Sequential, models
from keras.optimizers import RMSprop

# inspired by https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py and https://towardsdatascience.com/yet-another-text-generation-project-5cfb59b26255


def sample(our_preds, our_temp=1.0):
    our_preds = np.asarray(our_preds).astype('float64')
    our_preds = np.log(our_preds) / our_temp
    exp_preds = np.exp(our_preds)
    our_preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, our_preds, 1)

    return np.argmax(probas)


TOTAL_EPOCHS = 40
BATCH_SIZE = 512
LAYER_SIZE = 256

text = open('data/data_selected', 'r').read().lower()
print('Corpus length:', len(text))

max_sequence_len = 50
step = 3

extracted_sentences = []
target_chars = []

for i in range(0, len(text) - max_sequence_len, step):
    next_sentence = text[i: i + max_sequence_len]
    extracted_sentences.append(next_sentence)
    next_character_in_text = text[i + max_sequence_len]
    target_chars.append(next_character_in_text)

unique_chars = sorted(list(set(text)))
unique_char_indices = dict((char, unique_chars.index(char)) for char in unique_chars)
print('Number of sequences:', len(extracted_sentences))
print('Unique characters:', len(unique_chars))

x = np.zeros((len(extracted_sentences), max_sequence_len, len(unique_chars)), dtype=np.bool)
y = np.zeros((len(extracted_sentences), len(unique_chars)), dtype=np.bool)

for i, sentence in enumerate(extracted_sentences):
    for t, char in enumerate(sentence):
        x[i, t, unique_char_indices[char]] = 1
    y[i, unique_char_indices[target_chars[i]]] = 1

model = Sequential()
model.add(layers.LSTM(LAYER_SIZE, input_shape=(max_sequence_len, len(unique_chars))))
model.add(layers.Dense(len(unique_chars), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))

for epoch in range(1, TOTAL_EPOCHS):
    print('epoch', epoch)
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=1)

    # the following can be used to generate examples

    # random_start_index = random.randint(0, len(text) - max_sequence_len - 1)
    # generated_text = text[random_start_index: random_start_index + max_sequence_len]
    # print('--- Generating with seed: "' + generated_text + '"')
    #
    # for temperature in [0.2, 0.4, 0.6, 0.8]:
    #     print('------ temperature:', temperature)
    #     sys.stdout.write(generated_text)
    #
    #     for i in range(250):
    #         sampled = np.zeros((1, max_sequence_len, len(unique_chars)))
    #         for t, char in enumerate(generated_text):
    #             sampled[0, t, unique_char_indices[char]] = 1.
    #
    #         preds = model.predict(sampled, verbose=0)[0]
    #         next_index = sample(preds, temperature)
    #         next_char = unique_chars[next_index]
    #
    #         generated_text += next_char
    #         generated_text = generated_text[1:]
    #
    #         sys.stdout.write(next_char)
    #         sys.stdout.flush()
    #     print()
