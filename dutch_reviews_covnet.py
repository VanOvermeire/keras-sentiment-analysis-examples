from keras import models
from keras import layers
from keras.optimizers import RMSprop

from functions import data_preparer

EPOCHS = 20
TEXT_LENGTH = 500

TEN_THOUSAND = 10_000
TRAINING_LIMIT = 20_000
VALIDATION_LIMIT = 32_000

data, labels, tokenizer = data_preparer.get_data_as_padded_sequences(TEN_THOUSAND, TEXT_LENGTH)

training_data = data[:VALIDATION_LIMIT]
training_labels = labels[:VALIDATION_LIMIT]

model = models.Sequential()
model.add(layers.Embedding(TEN_THOUSAND, 128, input_length=TEXT_LENGTH))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

print('Compiling and fitting model')
model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(training_data, training_labels,
                    epochs=EPOCHS,
                    batch_size=256,
                    validation_split=0.2)
