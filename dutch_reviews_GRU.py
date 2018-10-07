from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from functions import data_preparer

EPOCHS = 15
TEXT_LENGTH = 500
TEN_THOUSAND = 10_000
TRAINING_LIMIT = 20_000
VALIDATION_LIMIT = 32_000

data, labels, tokenizer = data_preparer.get_data_as_padded_sequences(TEN_THOUSAND, TEXT_LENGTH)

training_data = data[:VALIDATION_LIMIT]
training_labels = labels[:VALIDATION_LIMIT]

evaluation_data = data[VALIDATION_LIMIT:]
evaluation_labels = labels[VALIDATION_LIMIT:]

callbacks = [EarlyStopping(monitor='acc', patience=2),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)]

model = models.Sequential()
model.add(layers.Embedding(TEN_THOUSAND, 32))
model.add(layers.GRU(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(layers.GRU(32, dropout=0.3, recurrent_dropout=0.2))
model.add(layers.Dense(1, activation='sigmoid'))

print('Compiling and fitting model')
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(training_data, training_labels,
                    epochs=EPOCHS,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks)

print('Testing')
results = model.evaluate(evaluation_data, evaluation_labels)
